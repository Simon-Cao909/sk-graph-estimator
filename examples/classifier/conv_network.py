from sk_graph_estimator.classifier import SKGraphClassifier
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32').reshape(x_train.shape[0],28,28,1) / 255
x_test = x_test.astype('float32').reshape(x_test.shape[0],28,28,1) / 255

model = SKGraphClassifier(model_structure = [
                                                {'type':'C', 'filters':8, 'kernel_size':(3,3), 'activation':'relu'},
                                                {'type':'MP'},

                                                {'type':'C', 'filters':16, 'kernel_size':(3,3), 'activation':'relu'},
                                                {'type':'MP'},

                                                {'type':'C', 'filters':32, 'kernel_size':(3,3), 'activation':'relu'},
                                                {'type':'MP'},

                                                {'type':'F'},
                                                {'type':'D', 'units':32, 'activation':'relu'},
                                                {'type':'D', 'units':10, 'activation':'softmax'}
                                            ],
                          epochs = 20,
                          learning_rate = 1e-3,
                          loss = 'categorical_crossentropy',
                          optimizer = 'adam',
                          batch_size = 128,
                          verbose = 1,)

model.fit(x_train,y_train)

print("Accuracy score:", model.score(x_test,y_test))