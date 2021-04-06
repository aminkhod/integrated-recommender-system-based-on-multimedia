
# coding: utf-8

# In[2]:


from snf import datasets
import numpy as np
import pandas as pd
import snf
from snf import compute
import sklearn


# In[3]:


digits = datasets.load_digits()
digits.keys()


# In[4]:


for arr in digits.data:
    print(arr.shape)


# In[5]:


groups, samples = np.unique(digits.labels, return_counts=True)


# In[6]:


for grp, count in zip(groups, samples):
    print('Group {:.0f}: {} samples'.format(grp, count))


# In[43]:


chemstracture = pd.read_csv('data/Drug disease/drug sim/Similarity_Matrix_Drugs chem stracture.txt',sep=' ', header=None)
# Phenotype = Phenotype.drop([1627],axis=1)
chemstracture = chemstracture.dropna(axis=1)
print(chemstracture.head())


sideeffect = pd.read_csv('data/Drug disease/drug sim/mat_drug_side effect.txt',sep=' ', header=None)
print(sideeffect.head())
sideeffect = sideeffect.replace(np.nan,0)

protein = pd.read_csv('data/Drug disease/drug sim/mat_drug_protein.txt',sep=' ', header=None)
print(protein.shape)


# In[44]:


# Phenotype = pd.read_csv('data/Drug disease/disease sim/mat_disease Phenotype.txt',sep='\t', header=None)
# Phenotype = Phenotype.drop([1627],axis=1)
# print(Phenotype.head())
# protein = pd.read_csv('data/Drug disease/disease sim/mat_disease_protein.txt',sep='\t', header=None)
# print(protein.head())


# In[45]:



# print(offsidesF.isnull().sum().sum())
protein = np.array(protein.values, dtype=np.float64)
chemstracture = np.array(chemstracture.values, dtype=np.float64)
sideeffect = np.array(sideeffect.values, dtype=np.float64)

proteinSim = sklearn.metrics.pairwise.cosine_similarity(protein, Y=None, dense_output=True)
chemstractureSim = sklearn.metrics.pairwise.cosine_similarity(chemstracture, Y=None, dense_output=True)
sideeffectSim = sklearn.metrics.pairwise.cosine_similarity(protein, Y=None, dense_output=True)
data = [chemstractureSim, proteinSim, sideeffectSim]
data


# In[46]:


# affinity_networks = compute.make_affinity(data, metric='cosine', K=9150, mu=0.5,normalize=True)
# # simcosine = pd.DataFrame(affinity_networks)
# # simcosine.to_csv("cosineSim.csv",index=False)


# In[47]:


# len(affinity_networks[1])


# In[48]:


# chem =  np.array(pd.read_csv('chem_Jacarrd_sim.csv', header=-1))
# enzyme =  np.array(pd.read_csv('enzyme_Jacarrd_sim.csv', header=-1))
# # chem = array(chem,enzyme)
# affinity_network = [chem,enzyme]


# In[49]:


fused_network = snf.snf(data, K=20)


# In[50]:


fused_network.shape


# In[51]:


fused_network


# In[52]:


# digits.data


# In[53]:


fused_network = pd.DataFrame(fused_network)
fused_network.to_csv('CosineSNF(chemstractureSim_proteinSim_sideeffectSim).csv',index = False,header=False)

