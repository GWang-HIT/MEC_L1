import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()
import xlrd2
import copy
from itertools import accumulate
np.set_printoptions(suppress=True)

#%% Read data from the Excel file
work_book = xlrd2.open_workbook('Data/L1_ePhys_20230608.xlsx')
table = work_book.sheet_by_name('Cell_175')
data_type = table.col_values(4)

data_type.pop(0)  # remove header line
print(data_type)
n = len(data_type)
print(n)
color_list = [0] * n
marker_list  = [0] * n
num_list  = [0] * n

for i in range(n):
    if data_type[i] == 'eNGFC':
        color_list[i] = '#FFBB00'
    elif data_type[i] == 'uNGFC':
        color_list[i] = '#0E86FF'
    elif data_type[i] == 'SBC':
        color_list[i] = '#FF0000'
        
for i in range(n):
    if data_type[i] == 'eNGFC':
        marker_list[i] = '*'
    elif data_type[i] == 'uNGFC':
        marker_list[i] = '++++++'
    elif data_type[i] == 'SBC':
        marker_list[i] = '000'
        
for i in range(n):
    if data_type[i] == 'eNGFC':
        num_list[i] = 1
    elif data_type[i] == 'uNGFC':
        num_list[i] = 0
    elif data_type[i] == 'SBC':
        num_list[i] = 2
        
n_uNGFC = 0
n_eNGFC = 0
n_SBC = 0
for i in range(n):
    if data_type[i] == 'eNGFC':
        n_eNGFC += 1
    elif data_type[i] == 'uNGFC':
        n_uNGFC += 1
    elif data_type[i] == 'SBC':
        n_SBC += 1
        
print(n_uNGFC,n_eNGFC,n_SBC)
print('Totol neurons number: %d' % len(data_type))

#%% Prepare data
number = 21
toldata = np.zeros((number, n))
for l in range(number):
    col_data = table.col_values(l + 7)
    col_data.pop(0)
    toldata[l] = np.array(col_data[0:n])
toldata = toldata.T

X = copy.deepcopy(toldata)
lab = copy.deepcopy(color_list)

# Z-score normalization
from sklearn import preprocessing
X_scale = preprocessing.scale(X)

#%% PCA & Clustering
feature_flag_list = []
k = 219564 # 6

feature_flag = bin(k)
feature_flag_list += [feature_flag]
m = len(feature_flag)
ind_feature = []
for j in range(m):
    if feature_flag[-j-1]=='1':
        ind_feature += [j]

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
pca = PCA(random_state=0).fit(X_scale[:,ind_feature])
X_pca = pca.transform(X_scale[:,ind_feature])
print(pca.explained_variance_ratio_)
print(list(accumulate(pca.explained_variance_ratio_)))

ind_col = list(range(6)) 
X_pca_slct = X_pca[:,ind_col]

agglomerative = AgglomerativeClustering(3)
labels_agg = agglomerative.fit_predict(X_pca_slct)
cluster_size = np.bincount(labels_agg)
print("Cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))

#%% Proportion
n_Cluster0_uNGFC = sum(np.where(np.array(data_type)=='uNGFC',1,0) &
                       np.where(labels_agg==0,1,0))
n_Cluster0_eNGFC = sum(np.where(np.array(data_type)=='eNGFC',1,0) &
                       np.where(labels_agg==0,1,0))
n_Cluster0_SBC = sum(np.where(np.array(data_type)=='SBC',1,0) &
                     np.where(labels_agg==0,1,0))

n_Cluster1_uNGFC = sum(np.where(np.array(data_type)=='uNGFC',1,0) &
                       np.where(labels_agg==1,1,0))
n_Cluster1_eNGFC = sum(np.where(np.array(data_type)=='eNGFC',1,0) &
                       np.where(labels_agg==1,1,0))
n_Cluster1_SBC = sum(np.where(np.array(data_type)=='SBC',1,0) &
                     np.where(labels_agg==1,1,0))

n_Cluster2_uNGFC = sum(np.where(np.array(data_type)=='uNGFC',1,0) &
                       np.where(labels_agg==2,1,0))
n_Cluster2_eNGFC = sum(np.where(np.array(data_type)=='eNGFC',1,0) &
                       np.where(labels_agg==2,1,0))
n_Cluster2_SBC = sum(np.where(np.array(data_type)=='SBC',1,0) &
                     np.where(labels_agg==2,1,0))

print(n_Cluster0_uNGFC/cluster_size[0], n_Cluster0_eNGFC/cluster_size[0], 
      n_Cluster0_SBC/cluster_size[0])
print(n_Cluster1_uNGFC/cluster_size[1], n_Cluster1_eNGFC/cluster_size[1], 
      n_Cluster1_SBC/cluster_size[1])
print(n_Cluster2_uNGFC/cluster_size[2], n_Cluster2_eNGFC/cluster_size[2], 
      n_Cluster2_SBC/cluster_size[2])

#%% Dendrogram
from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward(X_pca_slct)
plt.figure(figsize=(5, 5))
dendrogram(linkage_array, labels=marker_list, leaf_rotation=0, color_threshold=0, leaf_font_size=5,orientation='left')
plt.savefig('Dendrogram.svg')

#%% Scatter 3D
fig = plt.figure()
name = data_type
ax2 = Axes3D(fig,azim=-145,elev=18)
ax2.grid(True)
ax2.grid(color='b', alpha=0.6)
ax2.set_facecolor('none')
ax2.scatter(X[:, 3],X[:, 8],X[:, 12], s=30, c=lab, marker="o" , alpha=1)
ax2.set_facecolor('none')

ax2.set_xlabel('$τ$(ms)')
ax2.set_xlim(0,24)
ax2.set_xticks([0,4,8,12,16,20,24])
ax2.set_ylabel('AP latency (ms)')
ax2.set_ylim(0,600)
ax2.set_yticks([0,100,200,300,400,500,600])
ax2.set_zlabel('AHP FWHM')
ax2.set_zlim(0,300)
ax2.set_zticks([0,50,100,150,200,250,300])
#ax2.grid(None)
plt.savefig('Scatter.svg')
plt.show()