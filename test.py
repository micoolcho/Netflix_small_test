# Test for Matrix Factorization for Netflix
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

L = len(training_dataset)
print(L)

testing_dataframe = pandas.read_csv("ts.csv", header=None)
testing_dataset = testing_dataframe.values

X_test = testing_dataset[:,0:2]
Y_test = testing_dataset[:,2]

numUsers = len(numpy.unique(Users))
print(numUsers)
numMovies = len(numpy.unique(Movies))
print(numMovies)
print(Movies.shape)

model_user = Sequential()
model_user.add(Embedding(numUsers, 30, input_length=1))
model_user.add(Reshape(target_shape=(30,)))

model_movie = Sequential()
model_movie.add(Embedding(numMovies, 30, input_length=1))
model_movie.add(Reshape(target_shape=(30,)))

model = Sequential()
model.add(Merge([model_user, model_movie], mode='concat'))
model.add(Dense(64, activation='relu', init='uniform'))
model.add(Dense(64, activation='relu', init='uniform'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adamax', metrics=['accuracy'])
print(model.summary())

callbacks = []
# callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('movie_weights.h5', save_best_only=True)]
model.fit([Users.reshape((L,1)), Movies.reshape((L,1))], Ratings.reshape((L,1)), batch_size=1000, nb_epoch=50, validation_split=.1, callbacks=callbacks, verbose=2)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



# movie_count = 17771
# user_count = 2649430
# model_left = Sequential()
# model_left.add(Embedding(movie_count, 60, input_length=1))
# model_right = Sequential()
# model_right.add(Embedding(user_count, 20, input_length=1))
# model = Sequential()
# model.add(Merge([model_left, model_right], mode='concat'))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('sigmoid'))
# model.add(Dense(64))
# model.add(Activation('sigmoid'))
# model.add(Dense(64))
# model.add(Activation('sigmoid'))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adadelta')
# model.fit([tr[:,0].reshape((L,1)), tr[:,1].reshape((L,1))], tr[:,2].reshape((L,1)), batch_size=24000, nb_epoch=42, validation_data=([ ts[:,0].reshape((M,1)), ts[:,0].reshape((M,1))], ts[:,2].reshape((M,1))))
