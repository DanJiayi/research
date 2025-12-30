import torch
from torch.utils.data import Dataset, DataLoader
import sys, os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamic_net import CVRNet_Binary,TR_Binary
from causalml.metrics import plot_gain,plot_qini
from sklift.metrics import uplift_auc_score, qini_auc_score
import matplotlib
matplotlib.use('Agg')  # Safe for headless environments
import matplotlib.pyplot as plt


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



# ===============================
#   1. Setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
#   2. Load checkpoints
# ===============================
# ckpt_path1 = '/root/test01/research/CausalCVR/logs/criteo/checkpoints/test_MyNet_tr_ckpt_5_8_1e-05_1e-05_-0.0436_0.0319_-0.1106_0.0599.pth.tar'
# ckpt_path2 = '/root/test01/research/CausalCVR/logs/criteo/checkpoints/test_MyNet_ckpt_40_8_1e-06_1e-07_-0.0454_0.0100_-0.1151_0.0192.pth.tar'
ckpt_path1 = '/root/test01/research/CausalCVR/logs/criteo/checkpoints/CVRNet_tr_ckpt.pth.tar'
ckpt_path2 = '/root/test01/research/CausalCVR/logs/criteo/checkpoints/CVRNet_ckpt.pth.tar'

checkpoint1 = torch.load(ckpt_path1, map_location=device)
checkpoint2 = torch.load(ckpt_path2, map_location=device)

h = 8
cfg_density = [(12, h, 1, 'relu'), (h, h, 1, 'relu')]
cfg = [(h, h, 1, 'relu'), (h, 1, 1, 'id')]

model1 = CVRNet_Binary(cfg_density, cfg).to(device)
model1.load_state_dict(checkpoint1['model_state_dict'])
model1.to(device).eval()

model2 = CVRNet_Binary(cfg_density, cfg).to(device)
model2.load_state_dict(checkpoint2['model_state_dict'])
model2.to(device).eval()

print(f"Loaded model1 (Proposed) from {ckpt_path1}")
print(f"Loaded model2 (Ablation) from {ckpt_path2}")

# ===============================
#   3. Load TargetReg (optional)
# ===============================
if checkpoint1.get('TR_state_dict') and checkpoint1['TR_state_dict'][0] is not None:
    TargetReg1, TargetReg2 = TR_Binary().to(device), TR_Binary().to(device)
    TargetReg1.load_state_dict(checkpoint1['TR_state_dict'][0])
    TargetReg2.load_state_dict(checkpoint1['TR_state_dict'][1])
    TargetReg1.eval()
    TargetReg2.eval()
    print('Loaded TargetReg')
else:
    TargetReg1 = TargetReg2 = None

# ===============================
#   4. Load test data
# ===============================
load_path = '/root/test01/research/CausalCVR/dataset/criteo'
print('Loading data...')
data = pd.read_csv(load_path + '/test.csv')
test_matrix = torch.from_numpy(data.to_numpy()).float().to(device)

# ===============================
#   5. Helper functions
# ===============================
def get_iter(data_matrix, batch_size=1024, shuffle=False):
    dataset = Dataset_from_matrix(data_matrix)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_preds_eval(model, test_matrix, targetreg1=None, targetreg2=None, batch_size=1024):
    def h(t, pi):
        t = t.view(-1, 1)
        pi = pi.view(-1, 1)
        return t / (pi + 1e-9) - (1 - t) / (1 - pi + 1e-9)

    model.eval()
    preds1, preds2, ys1, ys2, ts = [], [], [], [], []
    test_loader = get_iter(test_matrix, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
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
                pred1 = torch.flatten(out[1][1] + trg1 * h(torch.ones_like(t), out[0])
                                      - out[1][0] - trg1 * h(torch.zeros_like(t), out[0]))
                pred2 = torch.flatten(out[2][1] + trg2 * h(torch.ones_like(t), out[0])
                                      - out[2][0] - trg2 * h(torch.zeros_like(t), out[0]))
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
    return y1, y2, preds1, preds2, t

# def compute_qini(df, outcome_col='y', treatment_col='treatment', score_col='score'):
#     y = df[outcome_col].values
#     t = df[treatment_col].values
#     s = df[score_col].values
#     order = np.argsort(-s)
#     y, t = y[order], t[order]
#     n_treat = (t == 1).sum()
#     n_ctrl = (t == 0).sum()
#     cum_treat = np.cumsum(y * (t == 1)) / (n_treat + 1e-9)
#     cum_ctrl = np.cumsum(y * (t == 0)) / (n_ctrl + 1e-9)
#     qini = cum_treat - cum_ctrl
#     return np.linspace(0, 1, len(qini)), qini

def compute_qini(df, outcome_col='y', treatment_col='treatment', score_col='score'):
    """
    Compute QINI curve strictly following formula:

    QINI(p) = NT(p)/NT  -  NC(p)/NC * NT(p)/(NT(p)+NC(p))
    """

    y = df[outcome_col].values
    t = df[treatment_col].values
    s = df[score_col].values

    # sort by predicted uplift score descending
    order = np.argsort(-s)
    y, t = y[order], t[order]

    # global totals
    NT = (t == 1).sum()
    NC = (t == 0).sum()

    # cumulative conversions in sorted list
    NTp = np.cumsum(y * (t == 1))      # conversions among treated up to p
    NCp = np.cumsum(y * (t == 0))      # conversions among control up to p

    # ======== QINI formula =========
    # QINI(p) = NT(p)/NT  -  NC(p)/NC * NT(p)/(NT(p)+NC(p))
    eps = 1e-9
    qini_curve = NTp / (NT + eps) - (NCp / (NC + eps)) * (NTp / (NTp + NCp + eps))

    # x-axis coverage proportion
    x = np.linspace(0, 1, len(qini_curve))
    return x, qini_curve

# def compute_uplift_curve(df, outcome_col='y', treatment_col='treatment', score_col='score'):
#     """
#     Fully reproduce sklift uplift_auc_score
#     while returning (x, curve, auuc) for plotting.
#     """

#     y = df[outcome_col].values
#     t = df[treatment_col].values
#     s = df[score_col].values

#     # sort by predicted uplift score desc
#     order = np.argsort(-s)
#     y, t = y[order], t[order]

#     Nt = (t == 1).sum()
#     Nc = (t == 0).sum()
#     r = Nt / Nc                                         # control reweight factor

#     cum_t = np.cumsum(y * (t == 1))
#     cum_c = np.cumsum(y * (t == 0))

#     # ----- uplift incremental gain curve (same as sklift implementation) -----
#     gain_curve = cum_t - r * cum_c

#     N = len(gain_curve)
#     x = np.arange(1, N+1) / N                           # population_frac

#     # ------ key: normalize by Nt instead of N  ------
#     auuc = np.trapz(gain_curve, x) / Nt                 # <--- FINAL CORRECT FIX
#     return x, gain_curve, auuc

def compute_gain_curve(df, outcome_col='y', treatment_col='treatment', score_col='score'):
    """
    Reproduce sklift.metrics.uplift_auc_score fully,
    while returning x-axis and gain curve for plotting.

    - curve matches incremental uplift gain curve
    - area under curve == uplift_auc_score (up to tiny float diff)

    returns:
        x: normalized population share
        gain_curve: uplift gain at each point (same shape as sklift)
        auuc: uplift_auc_score equivalent area
    """

    y = df[outcome_col].values
    t = df[treatment_col].values
    s = df[score_col].values

    # descending sort by uplift score
    order = np.argsort(-s)
    y, t = y[order], t[order]

    # total counts
    N_treat = (t == 1).sum()
    N_ctrl  = (t == 0).sum()
    r = N_treat / N_ctrl  # reweight factor

    # cumulative conversions
    cum_treat = np.cumsum(y * (t == 1))
    cum_ctrl  = np.cumsum(y * (t == 0))

    # === sklift gain curve definition ===
    #    gain[k] = cum_treat[k] - r*cum_ctrl[k]
    gain_curve = cum_treat - r * cum_ctrl

    # x-axis in percent of population
    x = np.arange(1, len(gain_curve)+1) / len(gain_curve)

    # area = uplift_auc_score equivalent
    # gain = np.trapz(gain_curve, x)

    return x, gain_curve

# ===============================
#   6. Predict both models
# ===============================
print('Predicting...')
y1, y2, preds1, preds2, t = get_preds_eval(model1, test_matrix, TargetReg1, TargetReg2)
_, _, preds1_b, preds2_b, _ = get_preds_eval(model2, test_matrix)

mask = (y1 == 1)

# ===============================
#   7. Compute AUUC & QINI metrics
# ===============================
auuc = uplift_auc_score(y2[mask], preds2[mask], t[mask])
qini = qini_auc_score(y2[mask], preds2[mask], t[mask])

auuc_b = uplift_auc_score(y2[mask], preds2_b[mask], t[mask])
qini_b = qini_auc_score(y2[mask], preds2_b[mask], t[mask])

print(f"AUUC_CVR (Proposed): {auuc:.5f}, AUUC_CVR (Ablation): {auuc_b:.5f}")
print(f"QINI_CVR (Proposed): {qini:.5f}, QINI_CVR (Ablation): {qini_b:.5f}")

# ===============================
#   8. Plot Qini comparison
# ===============================
plt.style.use('default')  
df_cvr_qini_a = pd.DataFrame({'y': y2[mask], 'treatment': t[mask], 'score': preds2[mask]})
df_cvr_qini_b = pd.DataFrame({'y': y2[mask], 'treatment': t[mask], 'score': preds2_b[mask]})

# x1, qini_curve1 = compute_qini(df_cvr_qini_a)
# x2, qini_curve2 = compute_qini(df_cvr_qini_b)

x1, qini_curve1 = compute_gain_curve(df_cvr_qini_a)
x2, qini_curve2 = compute_gain_curve(df_cvr_qini_b)

plt.figure(figsize=(8, 6))
plt.grid(False)
plt.plot(x1, qini_curve1, label='Proposed Method', color='#1f77b4', linewidth=2.2)
plt.plot(x2, qini_curve2, label='Plug-in Estimator', color='#ff7f0e', linewidth=2.2)
plt.plot(x1, np.linspace(0, qini_curve1[-1], len(x1)), '--', color='gray', label='Random')

# plt.title('Qini Curve Comparison (CVR)', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('/root/test01/research/CausalCVR/logs/qini_cvr_compare_test2.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved Qini comparison curve → logs/qini_cvr_compare_test2.png")

