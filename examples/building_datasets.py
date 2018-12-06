import csv
import json
import os
import requests
import zipfile

from itertools import islice
from lightfm import LightFM
from lightfm.data import Dataset

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

ratings, book_features = get_data()

# print out the ratings
for line in islice(ratings, 2):
	print(json.dumps(line, indent=4))

# print out the book features
for line in islice(book_features, 1):
	print(json.dumps(line, indent=4))

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

print(repr(interactions))

# item_features matrix can also be created
item_features = dataset.build_item_features(((x['ISBN'], [x['Book-Author']])
											  for x in get_book_features()))

print(repr(item_features))

# build the model using the custom dataset
model = LightFM(loss='bpr')
model.fit(interactions, item_features=item_features)

"""
# compare the accuracy of BPR model with the custom dataset
epochs = 70
num_components = 32

bpr_duration = []
bpr_auc = []

# times for the BPR model
for epoch in range(epochs):
	start = time.time()
	model.fit_partial(train, epochs=1)
	bpr_duration.append(time.time() - start)
	bpr_auc.append(auc_score(model, test, train_interactions=train).mean())

# plot the results 
x = np.arange(epochs)
plt.plot(x, np.array(bpr_auc))
plt.legend(['BPR AUC'], loc='upper right')
plt.show()

# plot the fitting accuracy
x = np.arange(epochs)
plt.plot(x, np.array(bpr_duration))
plt.legend(['BPR duration'], loc='upper right')
plt.show()
"""
