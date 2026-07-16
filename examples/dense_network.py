from adv_keras_regressor.regressor import AdvKerasRegressor

model = AdvKerasRegressor(model_structure = [
                                                {'type':'D', 'units':128, 'activation':'relu'},
                                                {'type':'d', 'rate':0.2},
                                                {'type':'D', 'units':64, 'activation':'relu'},
                                                {'type':'D', 'units':1, 'activation':'linear'}
                                            ],
                          epochs = 200,
                          learning_rate = 1e-3)
