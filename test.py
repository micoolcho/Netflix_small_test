# Test for Matrix Factorization for Netflix Challenge
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping, ModelCheckpoint

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

training_dataframe = pandas.read_csv("tr.csv", header=None)
training_dataset = training_dataframe.values

Movies = training_dataset[:,0]
Users = training_dataset[:,1]
Ratings = training_dataset[:,2]

numUsers = len(numpy.unique(Users))
numMovies = len(numpy.unique(Movies))
print("No of unique users: " + str(numUsers))
print("No of unique movies: " + str(numMovies))

model_user = Sequential()
model_user.add(Embedding(numUsers, 30, input_length=1))
model_user.add(Reshape(target_shape=(30,)))

model_movie = Sequential()
model_movie.add(Embedding(numMovies, 30, input_length=1))
model_movie.add(Reshape(target_shape=(30,)))

model = Sequential()
model.add(Merge([model_user, model_movie], mode='concat'))
model.add(Dense(64, activation='relu', init='uniform'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adamax', metrics=['accuracy'])
print(model.summary())

callbacks = []
# callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('movie_weights.h5', save_best_only=True)]
model.fit([Users.reshape((-1,1)), Movies.reshape((-1,1))], Ratings.reshape((-1,1)), batch_size=100, nb_epoch=20, validation_split=.1, callbacks=callbacks, verbose=2)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
