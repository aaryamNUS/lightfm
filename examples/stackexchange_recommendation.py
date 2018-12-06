import numpy as np 
# import the model
from lightfm import LightFM 
from lightfm.datasets import fetch_stackexchange
from lightfm.evaluation import auc_score

data = fetch_stackexchange('crossvalidated', test_set_fraction=0.1, indicator_features=False, tag_features=True)

train = data['train']
test = data['test']

print('The dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      # getnnz() --> gets the count of explicitly-stored values (i.e. non-zero values)
      % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))

# set the number of threads; can increase this
# if more physical cores are available. However, MacOS systems 
# use a default value of 1 thread if OpenMP is not supported
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

# Try to fit a WARP model - this is generally the model with the best performance
model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components = NUM_COMPONENTS)

# run 3 epochs and time it
model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)

# compute and print the AUC score
train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

# We pass in the train interactions to exclude them from predictions.
# This is to simulate a recommender system where we do not
# re-recommend things the user has already interacted with in the train set.
test_auc = auc_score(model, test, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# set the biases to zero to rid of pre-estimated per-item biases
model.item_biases *= 0.0

test_auc = auc_score(model, test, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# StackExchange data comes with content information in the form of tags users apply to their questions:
item_features = data['item_features']
tag_labels = data['item_feature_labels']

print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

# We can employ these features (as opposed to an identity feature matrix in a pure CF model) to estimate a model
# which will generalize better to unseen examples
# It will simply use its representations of item features to infer representations of previously unseen questions

# define a new model instance
model = LightFM(loss='warp',item_alpha=ITEM_ALPHA,no_components=NUM_COMPONENTS)

# fit the hybrid model. This time pass in the item features matrix
model = model.fit(train, item_features=item_features,epochs=NUM_EPOCHS,num_threads=NUM_THREADS)

# evaluate the hybrid model on the training set
train_auc = auc_score(model, train, item_features=item_features,num_threads=NUM_THREADS).mean()
print('Hybrid training set AUC: %s' % train_auc)

# evaluate the hybrid model on the test set
test_auc = auc_score(model, test, item_features=item_features,num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)

def get_similar_tags(model, tag_id):
	# define similarity as the cosine of the angle between the latent vectors

	# normalize vectors to unit length
	tag_embeddings = (model.item_embeddings.T / np.linalg.norm(model.item_embeddings, axis=1)).T

	query_embedding = tag_embeddings[tag_id]
	similarity = np.dot(tag_embeddings, query_embedding)
	most_similar = np.argsort(-similarity)[1:4]

	return most_similar

for tag in (u'bayesian', u'regression', u'survival'):
	tag_id = tag_labels.tolist().index(tag)
	print('Most similar tags for %s: %s' % (tag_labels[tag_id], tag_labels[get_similar_tags(model, tag_id)]))
