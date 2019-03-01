import numpy as np
from sklearn.datasets import load_iris, load_digits

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pickle

import umap
#pip3 install umap-learn

digits = load_digits()
all_fcs = []

with open('all_data.pkl', 'rb') as handle:
    all_data = pickle.load(handle)

for mission in all_data:
    for step in mission:
        all_fcs.append(step['fc'][0])

all_fcs = np.array(all_fcs)
reducer = umap.UMAP()

u = reducer.fit_transform(all_fcs)
plt.scatter(u[:,0], u[:,1])
plt.title('umap...')
plt.show()

print("done")
