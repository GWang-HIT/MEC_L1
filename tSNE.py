import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import xlrd2
from itertools import accumulate

wb1 = xlrd2.open_workbook('Data/L1_ePhys_20230608.xlsx')
sheet1 = wb1.sheets()[0]

num_cell = np.array([215, 82, 51])
iNumTotal = list(accumulate(num_cell))
iNumTotal.insert(0, 0)
iNumTotal = np.array(iNumTotal)

cmap_list1 = [[0, 0.5, 1.0], [1.0, 0.7, 0.0], [1, 0., 0.0]]
colormap1 = mpl.colors.LinearSegmentedColormap.from_list('cmap', cmap_list1, 3)

data_list = []
for i in range(0, sheet1.nrows):
    row_list = sheet1.row_values(i)
    data_list.append(row_list)   
data1 = pd.DataFrame(data_list)

#%% TSNE
ind_col = list(range(7,28))

X1 = np.array(data1.iloc[1:iNumTotal[3]+1, ind_col])
y1 = np.array(data1.iloc[1:iNumTotal[3]+1, 5])          

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X1)
X1_norm = scaler.transform(X1)
    
from sklearn.manifold import TSNE
model = TSNE(random_state=8)
projected = model.fit_transform(X1_norm)

fig, axes = plt.subplots(2, 2, figsize=(9, 9))
ax = axes.ravel()

for i in range(3):
    ax[i].scatter(projected[iNumTotal[i]:iNumTotal[i+1], 0], 
                  projected[iNumTotal[i]:iNumTotal[i+1], 1], 
                  color=cmap_list1[i], edgecolor='none', alpha=0.6)
    ax[i].set(xlim=(-30, 30), ylim=(-30, 30))  

ax[3].scatter(projected[:, 0], projected[:, 1],
            c=y1, edgecolor='none', alpha=0.6,
            cmap=colormap1)
ax[3].set(xlim=(-30, 30), ylim=(-30, 30)) 
plt.savefig('tSNE.svg')