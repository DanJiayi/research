import json
import numpy as np
with open('result.json', 'r', encoding='utf-8') as f:
    data = json.load(f) 

for k,v in data.items():
    print(k)
    lst1,lst2 = np.array([i[0] for i in v]),np.array([i[1] for i in v])
    print('mean1: ',lst1.mean(),' std1: ',lst1.std(),' mean2: ',lst2.mean(),' std2: ',lst2.std())
    print()
