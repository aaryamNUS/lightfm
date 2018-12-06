import numpy as np 
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

# sample a few users and get their recommendations --> used to give an indication of the accuracy of the model
def sample_recommendation(model, data, user_ids):

	n_users, n_items = data['train'].shape

	for user_id in user_ids:
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		scores = model.predict(user_id, np.arange(n_items))
		top_items = data['item_labels'][np.argsort(-scores)]

		print("User %s" % user_id)
		print("     known positives:")

		for x in known_positives[:3]:
			print("          %s" % x)

		print("     Recommended:")

		for x in top_items[:3]:
			print("          %s" % x)


# downloads data and pre-processes it into sparse matrices
# prepares the sparse user-item matrices: 1 means user interacted with the product, 0 means user didn't
data = fetch_movielens(min_rating=5.0)

# check training and test matrices
print(repr(data['train']))
print(repr(data['test']))

# train the model with WARP loss function
model = LightFM(loss='warp')
model.fit(data['train'], epochs=100, num_threads=2)

# evaluate how well the model did against the training and test data - WARP loss function
print("Train precision using WARP loss function: %.2f" % precision_at_k(model, data['train'], k=5).mean())
print("Test precision using WARP loss function: %.2f" % precision_at_k(model, data['test'], k=5).mean())

# train the model with BPR loss function
model2 = LightFM(loss='bpr')
model2.fit(data['train'], epochs=100, num_threads=2)

# evaluate how well the model does using BPR loss function
print("Train precicision using BPR loss function: %.2f" % precision_at_k(model2, data['train'], k=5).mean())
print("Test precicision using BPR loss function: %.2f" % precision_at_k(model2, data['test'], k=5).mean())

# train the model with logistic loss function
model3 = LightFM(loss='logistic')
model3.fit(data['train'], epochs=100, num_threads=2)

# evaluate how well the model does using logistic loss function
print("Train precicision using logistic loss function: %.2f" % precision_at_k(model3, data['train'], k=5).mean())
print("Test precicision using logistic loss function: %.2f" % precision_at_k(model3, data['test'], k=5).mean())

# train the model with WARP-kOS loss function
model4 = LightFM(loss='warp-kos')
model4.fit(data['train'], epochs=100, num_threads=2)

# evaluate how well the model does using WARP-kOS loss function
print("Train precicision using WARP kOS loss function: %.2f" % precision_at_k(model4, data['train'], k=5).mean())
print("Test precicision using WARP kOS loss function: %.2f" % precision_at_k(model4, data['test'], k=5).mean())

sample_recommendation(model, data, [3, 6, 12, 123, 524, 677])
