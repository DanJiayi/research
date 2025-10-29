import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamic_net import MyNet_Binary,TR_Binary
from causalml.metrics import plot_gain,plot_qini
from sklift.metrics import uplift_auc_score, qini_auc_score

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

ckpt_path = '/root/test01/research/CausalCVR/logs/criteo/checkpoints/test_MyNet_tr_ckpt_5_8_1e-05_1e-05_-0.0436_0.0319_-0.1106_0.0599.pth.tar'
checkpoint = torch.load(ckpt_path, map_location='cpu')

h = 8
cfg_density = [(12, h, 1, 'relu'), (h, h, 1, 'relu')]
cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]
model = MyNet_Binary(cfg_density, cfg).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


print(f"loaded {checkpoint['model']}")

if checkpoint.get('TR_state_dict') and checkpoint['TR_state_dict'][0] is not None:
    TargetReg1, TargetReg2 = TR_Binary(), TR_Binary()  # 你定义的 TargetReg 类
    TargetReg1.load_state_dict(checkpoint['TR_state_dict'][0])
    TargetReg2.load_state_dict(checkpoint['TR_state_dict'][1])
    TargetReg1.eval()
    TargetReg2.eval()
    print('loaded TargetReg')
else:
    TargetReg1 = TargetReg2 = None



def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def get_preds_eval(model, test_matrix, targetreg1=None, targetreg2=None, batch_size=1024):
    def h(t, pi):
        t = t.view(-1, 1)
        pi = pi.view(-1, 1)
        return t / (pi + 1e-9) - (1 - t) / (1 - pi + 1e-9)

    was_training = model.training
    model.eval()

    preds1, preds2 = [], []
    ys1, ys2, ts = [], [], []

    test_loader = get_iter(test_matrix, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, exposure in test_loader:
            t = inputs[:, 12]
            x = inputs[:, :12]
            y1 = inputs[:, 14]
            y2 = inputs[:, 13]

            out = model(x)

            if targetreg1 is None:
                pred1 = torch.flatten(out[1][1] - out[1][0])
                pred2 = torch.flatten(out[2][1] - out[2][0])
            else:
                trg1, trg2 = targetreg1(x), targetreg2(x)
                pred1 = torch.flatten(
                    out[1][1]
                    + trg1 * h(torch.ones_like(t), out[0])
                    - out[1][0]
                    - trg1 * h(torch.zeros_like(t), out[0])
                )
                pred2 = torch.flatten(
                    out[2][1]
                    + trg2 * h(torch.ones_like(t), out[0])
                    - out[2][0]
                    - trg2 * h(torch.zeros_like(t), out[0])
                )

            preds1.append(pred1.cpu())
            preds2.append(pred2.cpu())
            ys1.append(y1.cpu())
            ys2.append(y2.cpu())
            ts.append(t.cpu())

    preds1 = torch.cat(preds1).numpy()
    preds2 = torch.cat(preds2).numpy()
    y1 = torch.cat(ys1).numpy()
    y2 = torch.cat(ys2).numpy()
    t = torch.cat(ts).numpy()

    if was_training:
        model.train()

    return y1, y2, preds1, preds2, t


load_path = '/root/test01/research/CausalCVR/dataset/criteo'
print('loading data')
data = pd.read_csv(load_path + '/test.csv')
test_matrix = torch.from_numpy(data.to_numpy()).float().to(device)

print('predicting')
y1, y2, preds1, preds2, t = get_preds_eval(model, test_matrix, TargetReg1, TargetReg2)

mask = (y1 == 1)
auuc1 = uplift_auc_score(y1, preds1, t)
qini1 = qini_auc_score(y1, preds1, t)
auuc2 = uplift_auc_score(y2[mask], preds2[mask], t[mask])
qini2 = qini_auc_score(y2[mask], preds2[mask], t[mask])
print(f"AUUC_CTR: {auuc1:.4f}, AUUC_CVR: {auuc2:.4f}")
print(f"QINI_CTR: {qini1:.4f}, QINI_CVR: {qini2:.4f}")

print('ploting')
# === (1) AUUC 曲线 ===
df_cvr_auuc = pd.DataFrame({'y': y2[mask], 'treatment': t[mask], 'score': preds2[mask]})
df_cvr_qini = df_cvr_auuc.copy()
plt.figure(figsize=(7, 5))
plot_gain(df_cvr_auuc, outcome_col='y', treatment_col='treatment')  # 默认使用 'score' 列
plt.title('AUUC Curve (CVR)')
plt.xlabel('Proportion of population targeted')
plt.ylabel('Gain')
plt.tight_layout()
plt.savefig('/root/test01/research/CausalCVR/logs/auuc_cvr.png', dpi=300, bbox_inches='tight')
print("Saved AUUC Curve → logs/auuc_cvr.png")
plt.close()

# === (2) Qini 曲线 ===
plt.figure(figsize=(7, 5))
plot_qini(df_cvr_qini, outcome_col='y', treatment_col='treatment')
plt.title('Qini Curve (CVR)')
plt.xlabel('Proportion of population targeted')
plt.ylabel('Qini')
plt.tight_layout()
plt.savefig('/root/test01/research/CausalCVR/logs/qini_cvr.png', dpi=300, bbox_inches='tight')
print("Saved Qini Curve → logs/qini_cvr.png")
plt.close()
