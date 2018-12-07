import csv
import json
import os
import requests
import time
import zipfile
import numpy as np

from itertools import islice
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.data import Dataset
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k

# download the Goodbooks-10k example dataset
def _download(url: str, dest_path: str):
	req = requests.get(url, stream=True)
	req.raise_for_status()

	with open(dest_path, "wb") as fd:
		for chunk in req.iter_content(chunk_size=2 ** 20):
			fd.write(chunk)

# retrieve the data
def get_data():
	ratings_url = ("http://www2.informatik.uni-freiburg.de/" "~cziegler/BX/BX-CSV-Dump.zip")

	if not os.path.exists("data"):
		os.makedirs("data")

		_download(ratings_url, "data/data.zip")

	with zipfile.ZipFile("data/data.zip") as archive:
		return (
			csv.DictReader(
				(x.decode("utf-8", "ignore") for x in archive.open("BX-Book-Ratings.csv")),
				delimiter=";",
				),
			csv.DictReader(
				(x.decode("utf-8", "ignore") for x in archive.open("BX-Books.csv")), delimiter=";"
			),
		)

# get ratings from the data
def get_ratings():
	return get_data()[0]

# get book features from the data
def get_book_features():
	return get_data()[1]

np.set_printoptions(threshold=np.nan)

# set the number of threads; can increase this
# if more physical cores are available. However, MacOS systems 
# use a default value of 1 thread if OpenMP is not supported
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

ratings, book_features = get_data()

# print out the ratings
#for line in islice(ratings, 2):
	#print(json.dumps(line, indent=4))

# print out the book features
#for line in islice(book_features, 1):
	#print(json.dumps(line, indent=4))

# create a dataset and build the ID mappings
dataset = Dataset()
dataset.fit((x['User-ID'] for x in get_ratings()),
			(x['ISBN'] for x in get_ratings()))

# query the dataset to check how many users and items (i.e. books) it knows
num_users, num_items = dataset.interactions_shape()
print('Num users : {}, num_items {}.'.format(num_users, num_items))

# add some item feature mappings, and creates a unique feature for each author
# NOTE: more item ids are fitted than usual, to make sure our mappings are complete
# even if there are items in the features dataset that are not in the interaction set
dataset.fit_partial(items=(x['ISBN'] for x in get_book_features()),
					item_features=(x['Book-Author'] for x in get_book_features()))

# build the interaction matrix which is a main input to the LightFM model
# it encodes the interactions between the users and the items
(interactions, weights) = dataset.build_interactions(((x['User-ID'], x['ISBN'])
													   for x in get_ratings()))

# item_features matrix can also be created
item_features = dataset.build_item_features(((x['ISBN'], [x['Book-Author']])
											  for x in get_book_features()))

# split the current dataset into a training and test dataset
train, test = random_train_test_split(interactions, test_percentage=0.001, random_state=None)

# build the model using the training dataset, notice the use of item_features as well, 
# this is a hybrid model
model = LightFM(loss='warp',item_alpha=ITEM_ALPHA,no_components=NUM_COMPONENTS)

# train the hybrid model on the training dataset
model.fit(train,item_features=item_features,epochs=NUM_EPOCHS,num_threads=1)
print('Model with WARP loss function fit successfully')

# evaluate the hybrid model on the test set
test_auc = auc_score(model, test, item_features=item_features,num_threads=1).mean()
print('Hybrid test set AUC (WARP): %s' % test_auc)



# build an optimised model using WARP-kOS loss function
model = LightFM(loss='warp-kos', item_alpha=ITEM_ALPHA,no_components=NUM_COMPONENTS)

# train the hybrid model on the training dataset
model.fit(train,item_features=item_features,epochs=NUM_EPOCHS,num_threads=1)
print('Model with warp-kOS loss function fit successfully')

#evaluate the hybrid model on the test set
test_auc = auc_score(model, test, item_features=item_features,num_threads=1).mean()
print('Hybrid test set AUC (WARP-kOS): %s' % test_auc)
