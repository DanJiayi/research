import json
import numpy as np
with open('result.json', 'r', encoding='utf-8') as f:
    data = json.load(f) 

for k,v in data.items():
    print(k)
    for model,mse in v.items():
        lst1,lst2 = np.array([i[0] for i in mse]),np.array([i[1] for i in mse])
        print('model: ',model,' len1: ',len(lst1),' mean1: ',lst1.mean(),' std1: ',lst1.std(),' len2: ',len(lst2),' mean2: ',lst2.mean(),' std2: ',lst2.std())
    print()