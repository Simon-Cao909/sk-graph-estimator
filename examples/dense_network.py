from adv_keras_regressor.regressor import AdvKerasRegressor
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

xtr_shape = x_train.shape
xte_shape = x_test.shape

x_train = x_train.reshape(xtr_shape[0],xtr_shape[1]*xtr_shape[2])
x_test = x_test.reshape(xte_shape[0],xte_shape[1]*xte_shape[2])

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

model = AdvKerasRegressor(model_structure = [
                                                {'type':'D', 'units':128, 'activation':'relu'},
                                                {'type':'d', 'rate':0.1},
                                                {'type':'D', 'units':64, 'activation':'relu'},
                                                {'type':'d', 'rate':0.1},
                                                {'type':'D', 'units':64, 'activation':'relu'},
                                                {'type':'d', 'rate':0.1},
                                                {'type':'D', 'units':10, 'activation':'softmax'}
                                            ],
                          epochs = 20,
                          learning_rate = 1e-3,
                          loss = 'categorical_crossentropy',
                          optimizer = 'adam',
                          batch_size = 128,
                          verbose = 1,)

model.fit(x_train,y_train)

print("R^2 score:", model.score(x_test,y_test))