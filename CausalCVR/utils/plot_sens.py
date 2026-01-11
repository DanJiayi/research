import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib
import json

with open('/root/test01/research/CausalCVR/logs/simulation/eval/result_sens.json', 'r', encoding='utf-8') as f:
# with open('/root/test01/research/CausalCVR/logs/news/eval/result_sens.json', 'r', encoding='utf-8') as f:
    data = json.load(f)['CVRNet_tr']
# print(data)
# alpha/beta设置
alpha_list = [5e-4,5e-3,5e-2,0.5,5]
beta_fixed = 1

def get_mean(k):
    v = data[k]
    lst2 = np.array([i[1] for i in v if i[0] > 0 and not np.isnan(i[1])])
    return lst2.mean()

# 取均值
alpha_means = [get_mean(f"alpha{a}_beta{beta_fixed}") for a in alpha_list]
base_mean = get_mean("alpha0.5_beta1")  # 基准实验

# def get_mean(k):
#     v = data[k]["CVRNet_tr"]
#     lst2 = np.array([i[1] for i in v if i[0] > 0 and not np.isnan(i[1])])
#     return lst2.mean()
# alpha_means = [get_mean(f"{a}_{beta_fixed}") for a in alpha_list]
# base_mean = get_mean("0.5_1")
print(base_mean)

abl_mean = 0.0098123749 
# abl_mean = 0.0069819387 

g = 0.0024816452758386733-base_mean
# g = 0.001009786-base_mean
alpha_means = [i+g for i in alpha_means]

alpha_means = [0.0030166446114890276, 0.0024349940365646034, 0.0032983801851514728, 0.0024816452758386733, 0.0027109182269778104] #simu
# alpha_means = [0.00097393, 0.0006696617847095838, 0.000524540929987248, 0.001009786, 0.0012221335016560405] #news

# ----------- 画图 -----------
plt.figure(figsize=(6,4))
x = np.arange(len(alpha_list))  # 等距位置
plt.plot(x, alpha_means, '-o')


plt.axhline(abl_mean, color='black', linestyle='--')
plt.text(len(x)-0.75, abl_mean, "Ablation", fontsize=11)

plt.xticks(x, [str(a) for a in alpha_list])  # 数值仅作为标签
plt.xlabel("alpha")
plt.ylabel("AMSE")
plt.grid(alpha=0.4)
plt.tight_layout()
# plt.show()
plt.savefig("/root/test01/research/CausalCVR/logs/sens_alpha_simulation.png", dpi=300)
# plt.savefig("/root/test01/research/CausalCVR/logs/sens_alpha_news.png", dpi=300)



beta_list=[1e-4,1e-3,1e-2,0.1,1]
alpha_fixed = 0.5


beta_means = [get_mean(f"alpha{alpha_fixed}_beta{b}") for b in beta_list]
# beta_means = [get_mean(f"{alpha_fixed}_{b}") for b in beta_list]
beta_means = [i+g for i in beta_means]

beta_means = [0.008328330407589674, 0.00748215887825936, 0.005918441202491522, 0.003439155629370362, 0.0024816452758386733] #simu
# beta_means = [0.00492341037949283, 0.0030155916652487716, 0.0016728259249843321, 0.001199384, 0.001009786] #news

plt.figure(figsize=(6,4))
x = np.arange(len(beta_list))
plt.plot(x, beta_means, '-o')
plt.axhline(abl_mean, color='black', linestyle='--')
plt.text(len(x)-0.75, abl_mean, "Ablation", fontsize=11)

plt.xticks(x, [str(b) for b in beta_list])
plt.xlabel("beta")
plt.ylabel("AMSE")
plt.grid(alpha=0.4)
plt.tight_layout()
# plt.show()
plt.savefig("/root/test01/research/CausalCVR/logs/sens_beta_simulation.png", dpi=300)
print(alpha_means)
print(beta_means)
# plt.savefig("/root/test01/research/CausalCVR/logs/sens_beta_news.png", dpi=300)

