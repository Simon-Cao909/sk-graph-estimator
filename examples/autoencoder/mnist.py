from sk_graph_estimator.autoencoder import SKGraphAutoencoder
from tensorflow import keras
import numpy as np

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32').reshape(x_train.shape[0],28,28,1) / 255
x_test = x_test.astype('float32').reshape(x_test.shape[0],28,28,1) / 255

x_train = x_train[:10000]
x_test = x_test[:5000]

model = SKGraphAutoencoder(encoder_structure=[
                                    {'type':'C','filters':16,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'MP','pool_size':(2,2),'strides':(2,2)},
                                    {'type':'C','filters':32,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'F'},
                                    {'type':'D','units':100,'activation':'linear'}
                            ],
                            decoder_structure=[
                                    {'type':'D','units':7*7*32,'activation':'relu'},
                                    {'type':'custom','layer':keras.layers.Reshape((7,7,32))},
                                    {'type':'CT','filters':32,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'UP','size':(2,2)},
                                    {'type':'CT','filters':16,'kernel_size':(3,3),'activation':'relu','padding':'same'},
                                    {'type':'UP','size':(2,2)},
                                    {'type':'CT','filters':1,'kernel_size':(3,3),'activation':'sigmoid','padding':'same'},
                            ],
                            model_type='standard', # or variational
                            epochs=20,
                            batch_size=128,
                            verbose=1,
                            optimizer='adam',
)

model.fit(x_train)
pred = model.predict(x_test)