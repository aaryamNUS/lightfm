import numpy as np 
import matplotlib
import numpy as np
import matplotlib.pyplot as plt 

from lightfm import LightFM 
from lightfm.datasets import fetch_movielens as mov 
from lightfm.evaluation import auc_score

# retrieve the data and define the evaluations
movielens = mov()
train, test = movielens['train'], movielens['test']

# evaluate performance of both models using WARP loss function
alpha = 1e-3
epochs = 70

# instantiate both models
adagrad_model = LightFM(no_components=30,loss='warp',learning_schedule='adagrad',user_alpha=alpha,item_alpha=alpha)
adadelta_model = LightFM(no_components=30,loss='warp',learning_schedule='adadelta',user_alpha=alpha,item_alpha=alpha)

# evaluate adagrad model
adagrad_auc = []

for epoch in range(epochs):
	# fit_partial resumes training from the current model state
	adagrad_model.fit_partial(train, epochs=1)
	adagrad_auc.append(auc_score(adagrad_model, test).mean())

# evaluate adadelta model
adadelta_auc = []

for epoch in range(epochs):
	adadelta_model.fit_partial(train, epochs=1)
	adadelta_auc.append(auc_score(adadelta_model, test).mean())

# plot the results of training over each epoch
x = np.arange(len(adagrad_auc))
plt.plot(x, np.array(adagrad_auc))
plt.plot(x, np.array(adadelta_auc))
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()

"""
# evaluate performace of both learning methods using WARP-kOS loss function

# instantiate both models
adagrad_model = LightFM(no_components=30,loss='warp-kos',learning_schedule='adagrad',user_alpha=alpha,item_alpha=alpha)
adadelta_model = LightFM(no_components=30,loss='warp-kos',learning_schedule='adadelta',user_alpha=alpha,item_alpha=alpha)

# evaluate adagrad model
adagrad_auc = []

for epoch in range(epochs):
	# fit_partial resumes training from the current model state
	adagrad_model.fit_partial(train, epochs=1)
	adagrad_auc.append(auc_score(adagrad_model, test).mean())

# evaluate adadelta model
adadelta_auc = []

for epoch in range(epochs):
	adadelta_model.fit_partial(train, epochs=1)
	adadelta_auc.append(auc_score(adadelta_model, test).mean())

# plot the results of training over each epoch
x = np.arange(len(adagrad_auc))
plt.plot(x, np.array(adagrad_auc))
plt.plot(x, np.array(adadelta_auc))
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()
"""
"""
# evaluate performace of both learning methods using BPR loss function

# instantiate both models
adagrad_model = LightFM(no_components=30,loss='bpr',learning_schedule='adagrad',user_alpha=alpha,item_alpha=alpha)
adadelta_model = LightFM(no_components=30,loss='bpr',learning_schedule='adadelta',user_alpha=alpha,item_alpha=alpha)

# evaluate adagrad model
adagrad_auc = []

for epoch in range(epochs):
	# fit_partial resumes training from the current model state
	adagrad_model.fit_partial(train, epochs=1)
	adagrad_auc.append(auc_score(adagrad_model, test).mean())

# evaluate adadelta model
adadelta_auc = []

for epoch in range(epochs):
	adadelta_model.fit_partial(train, epochs=1)
	adadelta_auc.append(auc_score(adadelta_model, test).mean())

# plot the results of training over each epoch
x = np.arange(len(adagrad_auc))
plt.plot(x, np.array(adagrad_auc))
plt.plot(x, np.array(adadelta_auc))
plt.legend(['adagrad', 'adadelta'], loc='lower right')
plt.show()
"""
