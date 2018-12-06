import numpy as np 
import time
import matplotlib
import matplotlib.pyplot as plt

from lightfm import LightFM 
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import auc_score

# fetch the data, using movielens 100K dataset
movielens = fetch_movielens()

train, test = movielens['train'], movielens['test']

# compare the accuracy of BPR and WARP models, and check how long each epoch takes
alpha = 1e-05
epochs = 70
num_components = 32

warp_model = LightFM(no_components=num_components, loss='warp-kos', learning_schedule='adagrad', max_sampled=100, user_alpha=alpha, item_alpha=alpha)

bpr_model = LightFM(no_components=num_components, loss='bpr', learning_schedule='adagrad', max_sampled=100, user_alpha=alpha, item_alpha=alpha)

warp_duration = []
bpr_duration = []
warp_auc = []
bpr_auc = []

# times for the WARP model
for epoch in range(epochs):
	start = time.time()
	warp_model.fit_partial(train, epochs=1)
	warp_duration.append(time.time() - start)
	warp_auc.append(auc_score(warp_model, test, train_interactions=train).mean())

# times for the BPR model
for epoch in range(epochs):
	start = time.time()
	bpr_model.fit_partial(train, epochs=1)
	bpr_duration.append(time.time() - start)
	bpr_auc.append(auc_score(bpr_model, test, train_interactions=train).mean())

# plot the results 
x = np.arange(epochs)
plt.plot(x, np.array(warp_auc))
plt.plot(x, np.array(bpr_auc))
plt.legend(['WARP-KOS AUC', 'BPR AUC'], loc='upper right')
plt.show()

# plot the fitting accuracy
x = np.arange(epochs)
plt.plot(x, np.array(warp_duration))
plt.plot(x, np.array(bpr_duration))
plt.legend(['WARP-KOS duration', 'BPR duration'], loc='upper right')
plt.show()
