import math
import numpy as np
import pandas as pd
import os
import json
from econml.grf import CausalForest
import argparse
from sklift.metrics import uplift_auc_score, qini_auc_score

data_path = '/root/test01/research/CausalCVR/dataset/criteo'
print('loading data')
train = pd.read_csv(data_path+'/train.csv')
test = pd.read_csv(data_path+'/test.csv')

features = [f"f{i}" for i in range(12)]
X_train, T_train = train[features].values, train["treatment"].values
Y1_train, Y2_train = train["visit"].values, train["conversion"].values

X_test, T_test = test[features].values, test["treatment"].values
Y1_test, Y2_test = test["visit"].values, test["conversion"].values

print('training model1')
model1 = CausalForest() #inference=False, fit_intercept=False
model1.fit(X_train, T_train, Y1_train)

print('training model2')
idx = np.where(Y1_train > 0)[0]
model2 = CausalForest() #inference=False, fit_intercept=False
model2.fit(X_train[idx], T_train[idx], Y2_train[idx])
uplift_pred_y1 = model1.predict(X_test)
uplift_pred_y2 = model2.predict(X_test)

print('evaluating model1')
uauc_y1 = uplift_auc_score(Y1_test, uplift_pred_y1, T_test)
qini_y1 = qini_auc_score(Y1_test, uplift_pred_y1, T_test)
print('evaluating model2')
mask = (Y1_test == 1)
uauc_y2 = uplift_auc_score(Y2_test[mask], uplift_pred_y2[mask], T_test[mask])
qini_y2 = qini_auc_score(Y2_test[mask], uplift_pred_y2[mask], T_test[mask])

print(f"Y1 uplift: AUUC={uauc_y1:.8f}, Qini={qini_y1:.8f}")
print(f"Y2 uplift: AUUC={uauc_y2:.8f}, Qini={qini_y2:.8f}")


# import numpy as np
# import pandas as pd
# from xgboost import XGBRegressor
# from sklift.metrics import uplift_auc_score, qini_auc_score

# # ---------- T-Learner 实现 ----------
# def fit_tlearner(X, T, Y):
#     """训练两个独立模型 (treatment=1, control=0)"""
#     model_t = XGBRegressor(
#         n_estimators=300,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )
#     model_c = XGBRegressor(
#         n_estimators=300,
#         max_depth=4,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     )

#     model_t.fit(X[T == 1], Y[T == 1])
#     model_c.fit(X[T == 0], Y[T == 0])
#     return model_t, model_c


# def predict_uplift(model_t, model_c, X):
#     """预测 uplift = E[Y|T=1,X] - E[Y|T=0,X]"""
#     mu1 = model_t.predict(X)
#     mu0 = model_c.predict(X)
#     return mu1 - mu0


# # ---------- 模型训练 ----------
# print('training model1 (Y1 ~ T + X)')
# model1_t, model1_c = fit_tlearner(X_train, T_train, Y1_train)

# print('training model2 (Y2 ~ T + X, only Y1>0 subset)')
# idx = np.where(Y1_train > 0)[0]
# model2_t, model2_c = fit_tlearner(X_train[idx], T_train[idx], Y2_train[idx])

# # ---------- uplift prediction ----------
# uplift_pred_y1 = predict_uplift(model1_t, model1_c, X_test)
# uplift_pred_y2 = predict_uplift(model2_t, model2_c, X_test)

# # ---------- evaluation ----------
# print('evaluating model1')
# uauc_y1 = uplift_auc_score(Y1_test, uplift_pred_y1, T_test)
# qini_y1 = qini_auc_score(Y1_test, uplift_pred_y1, T_test)

# print('evaluating model2 (only Y1_test==1 subset)')
# mask = (Y1_test == 1)
# uauc_y2 = uplift_auc_score(Y2_test[mask], uplift_pred_y2[mask], T_test[mask])
# qini_y2 = qini_auc_score(Y2_test[mask], uplift_pred_y2[mask], T_test[mask])

# print(f"Y1 uplift: AUUC={uauc_y1:.8f}, Qini={qini_y1:.8f}")
# print(f"Y2 uplift: AUUC={uauc_y2:.8f}, Qini={qini_y2:.8f}")
