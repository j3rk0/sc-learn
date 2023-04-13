import datatable as dt
from sclearn.normalization import TMMNormalizer
from sclearn.gene_selection import LowQualityGeneFilter
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sclearn.clustering import PhenoClust
from sklearn.metrics.cluster import rand_score
import numpy as np
#%%
data = dt.fread('data/ts_liver.csv').to_pandas()

#%%
labels = data['cell_type']
enc = LabelEncoder()
y = enc.fit_transform(labels)
data = data.loc[:,[c for c in data.columns if 'ENSG' in c]]

#%%
preprocessing_pipe = Pipeline([('qc',LowQualityGeneFilter()),('norm',TMMNormalizer())])
preprocessing_pipe.fit(data)
data = preprocessing_pipe.transform(data)

#%%
dimred = Pipeline([('linear',PCA(n_components=50)),('nonlinear',TSNE(n_components=2))])
toplot = dimred.fit_transform(data)

#%%
scatter = plt.scatter(toplot[:,0],toplot[:,1],s=1,c=y,cmap='tab20')
scatter.legend_elements(prop='colors', num=len(enc.classes_))
handles = scatter.legend_elements(prop="colors")[0]
plt.legend(handles=handles,labels=list(enc.classes_),loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

#%%
clust = PhenoClust(resolution=.5,metric='cityblock',k=20)
pred = clust.fit_predict(data)

print(f"rand-index: {rand_score(y,pred)}")

#%%
x1 = data[ np.array(labels == 'Erythrocyte'),:]
x2 = data[np.array(labels == 'Dendritic cell'),:]
x = np.concatenate([x1,x2])

#%%
from sklearn.svm import SVR

mu = x.mean(axis=0)

x = x[:,mu>0]
mu = mu[mu>0]
var = x.var(axis=0)
disp = (var-mu)/mu**2

lm = SVR(kernel='poly',degree=2)
lm.fit(mu.reshape(-1,1),disp)
est = lm.predict(mu.reshape(-1,1))

plt.scatter(mu,disp,c='black',s=1)
plt.scatter(mu,est,c='red',s=1)
plt.show()
