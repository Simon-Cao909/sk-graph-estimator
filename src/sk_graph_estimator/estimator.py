from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from scipy.sparse import issparse
from tensorflow import keras
from tensorflow.keras import layers as kl
import numpy as np

from .tools.sklearn_layer import SKlearnLayer
from .tools.check_shapes import shapes_equal
from .tools.quick_build_parser import parse_quick

class SKGraphEstimator(BaseEstimator):
    '''
    SKGraphEstimator is a machine learning algorithm that combines the user-friendly 
    features of scikit-learn regressors and the versatility of Tensorflow with Keras
    '''
    
    def __init__(
        self,
        model_structure,
        build_setting="normal",
        input_shape=None,
        epochs=100,
        batch_size=32,
        early_stopping=True,
        n_iter_no_change=10,
        validation_split=0.1,
        verbose=1,
        loss="mse",
        metrics=None,
        optimizer="adam",
        learning_rate=1e-3,
        random_state=None,
        shuffle=True,
    ):
        '''
        Attributes
        - model_structure (list or tuple): Specifies the model architecture
                                           See architecture.md for how to format this
        - build_setting (str, default="normal"): Decides the format of model_structure
                                                 Must be either 'normal' or 'quick'
                                                 See architecture.md for more information
        - input_shape (tuple, default=None): The input shape
                                             If None, it will be guessed from the feature shape
        - epochs (int, default=100): The number of epochs to train the model for
        - batch_size (int, default=32): The batch size for training
        - early_stopping (bool, default=True): Whether the model should stop training early if validation
                                               loss doesn't drop after n_iter_no_change iterations
        - n_iter_no_change (int, default=10): The amount of iterations without validation 
                                              loss change until the model stops training
                                              (only matters if early_stopping is True)
        - validation_split (float): Should be between 0 and 1
                                    This will determine how the training and validation data are split
                                    with validation_split being the fraction of validation data
        - verbose (int): If 0, nothing is printed. If 1, the process of training is printed
        - loss (str or callable or list, default='mse'): The loss function used. See Keras for custom ones
                                                        If your model has a multi-output layer, you can use
                                                        a list where the ith loss corresponds to the ith output
        - metrics (list, tuple, dict, or None, default=None): The metrics tracked during training
        - optimizer (str, default='adam'): The optimizer used in training. See Keras for possibilities
        - learning_rate (float, default=1e-4): The learning rate for training
        - random_state (int or None, default=None): The random state. Used for reproducible results
        - shuffle (bool, default=True): Whether to shuffle the data before training
        '''
        self.model_structure = model_structure
        self.build_setting = build_setting
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_split = validation_split
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.shuffle = shuffle


    ### HELPER FUNCTIONS ###

    def _make_optimizer(self):
        '''
        Creates the optimizer for the model

        :return (keras.Optimizer): The optimizer object
        '''
        if isinstance(self.optimizer, str):
            opt = keras.optimizers.get(self.optimizer)
        else:
            opt = keras.optimizers.deserialize(keras.optimizers.serialize(self.optimizer))

        if self.learning_rate is not None:
            opt.learning_rate = self.learning_rate

        return opt

    def _validate_hyperparams(self):
        '''
        Validates the hyperparameters of the model
        Will raise an error of they are not proper
        '''
        if not isinstance(self.model_structure,(list,tuple)):
            raise TypeError("model_structure must be a list or tuple")
        if len(self.model_structure) == 0:
            raise ValueError("model_structure cannot be empty")

        if self.build_setting == "normal":
            if any(not isinstance(struct,dict) for struct in self.model_structure):
                raise TypeError("Each struct in model_structure must be a dict")
            if any(struct.get('type') is None for struct in self.model_structure):
                raise KeyError("Each struct in model_structure must have key 'type'")
        
        if self.validation_split is not None and not (0 <= self.validation_split < 1):
            raise ValueError("validation_split must be in [0, 1).")
        
        if self.early_stopping and (self.validation_split is None or self.validation_split <= 0):
            raise ValueError("early_stopping=True requires validation_split > 0.")

    def _project(self,x,target_shape):
        '''
        Safely projects a given tensor onto a given target shape

        :param x (KerasTensor): The input tensor
        :param target_shape (tuple): The target shape

        :return (KerasTensor): The tensor after projection
        '''
        shape_len = len(target_shape)
        if shape_len == 2:
            return kl.Dense(target_shape[-1])(x)

        elif shape_len == 4:
            _, target_h, target_w, target_c = target_shape
            curr_w, curr_h = keras.backend.int_shape(x)[1:3]

            if curr_h % target_h != 0 or curr_w % target_w != 0:
                raise ValueError(f"Projection not possible from ({curr_h}, {curr_w}) ->"
                                 f" ({target_h,target_w})")
            
            stride_h = curr_h // target_h
            stride_w = curr_w // target_w
            
            return kl.Conv2D(filters=target_c,
                             kernel_size=(1,1),
                             strides=(stride_h,stride_w),
                             padding="same",
                             use_bias=False)(x)
        
        raise ValueError(f"Projection not supported for target_shape {target_shape}")

    def _get_callbacks(self):
        '''
        Creates and returns a list of callbacks
        Currently only supports early stopping

        :return (list): A list of callbacks
        '''
        callbacks = []

        if self.early_stopping:
            patience = self.n_iter_no_change
            if patience is None:
                patience = max(5, int(self.epochs * 0.1))

            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )
        
        return callbacks

    def _validate_data(self,X,y=None):
        '''
        Checks the data to see if it is of the proper format

        :param X (array-like): The feature array
        :param y (array-like, default=None): The labels array
                                             If None, only the features
                                             will be checked and returned

        :return (np.ndarray or tuple of ndarrays): If y was given, returns X and y as an array
                                                   Else, X will be returned as an array
        '''

        if self.is_multi_output_ and y is not None:
            if len(y) != len(self.output_shape_):
                raise ValueError(
                    f"Expected {len(self.output_shape_)} outputs, "
                    f"got {len(y)} outputs instead"
                )
            
            for i,(target,expec_shape) in enumerate(zip(y,self.output_shape_)):
                if target.shape[1:] != expec_shape:
                    raise ValueError(
                        f"For output {i}: expected shape {expec_shape}, "
                        f"got {target.shape[1:]} instead"
                    )
        elif len(self.output_shape_) <= 2 and len(self.input_shape_) <= 2:
            if y is None:
                X = check_array(X, accept_sparse=False, dtype=np.float32)
                X = validate_data(self,X,reset=False)
            else:
                X, y = validate_data(
                    self,
                    X,
                    y,
                    multi_output=True,
                    y_numeric=True,
                    dtype=np.float32,
                )
        
        if X.shape[1:] != self.input_shape_:
            raise ValueError(
                f"The features have {X.shape[1:]} shape, but this model was fitted with "
                f"{self.input_shape_} input shape."
            )
        
        if y is not None and not self.is_multi_output_ and self.output_shape_ != y.shape[1:]:
            raise ValueError(
                f"output_shape={self.output_shape_}, but y has shape {y.shape[1:]}"
            )
        
        if self.is_multi_output_:
            return np.asarray(X) if y is None else (np.asarray(X),y)
        
        return np.asarray(X) if y is None else (np.asarray(X), np.asarray(y))

    def _check_is_fitted(self):
        check_is_fitted(self, "model_")

        
    ### ADDING BLOCKS ###

    def _add_simple_block(self,layer_type,layer_specs,ind,x):
        '''
        Adds a layer to the model with the given hyperparameters

        :param layer_type (str): The layer type. See architecture.md for possibilities
        :param layer_specs (dict): A dictionary containing the hyperparameters
        :param ind (int or str): The layer index
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after the specified layer has been applied
        '''
        if layer_type == "D" or layer_type.lower() == 'dense':
            if not isinstance(layer_specs.get('units'),int):
                if isinstance(layer_specs.get('neurons'),int):
                    num_neurons = layer_specs['neurons']
                else:
                    raise ValueError(f"# of neurons must be integer, layer {ind}")
            else:
                num_neurons = layer_specs['units']
            if layer_specs.get('activation') is None:
                raise KeyError(f"No activation function given for layer {ind}")
            
            return kl.Dense(num_neurons, activation=layer_specs['activation'])(x)
        elif layer_type == 'd' or layer_type.lower() == 'dropout':
            if layer_specs.get('rate') is None:
                raise KeyError(f"No dropout rate given for layer {ind}")
            if not (0 <= layer_specs['rate'] < 1):
                raise ValueError(f"drop_out values must be in [0,1) for layer {ind}")
            
            return kl.Dropout(layer_specs['rate'])(x)
        elif layer_type in ['C','CT'] or layer_type.lower() in ['conv','convolution']+\
                                                               ['conv_transpose','convolution_transpose']:
            
            if layer_type == 'C' or layer_type.lower() in ['conv','convolution']:
                layer_type = 'C'
            else:
                layer_type = 'CT'

            if layer_specs.get('filters') is None:
                raise KeyError(f"Filters must be given for layer {ind}")
            if layer_specs.get('activation') is None:
                raise KeyError(f"No activation function given for layer {ind}")
            
            kernel_size = layer_specs.get('kernel_size')

            if kernel_size is None:
                raise KeyError(f"No kernel size given for convolutional layer {ind}")
            if not isinstance(kernel_size,tuple):
                raise ValueError(f"Layer {ind}: kernel_size must be tuple")

            conv_d = len(kernel_size)
            default_stride = tuple([1]*conv_d)
                        
            if len(x.shape) != conv_d+2:
                raise ValueError(f"Expected input to Conv{conv_d}D to have rank {conv_d+2}, got shape {x.shape}")

            if conv_d == 1:
                Conv = kl.Conv1D if layer_type == 'C' else kl.Conv1DTranspose
            elif conv_d == 2:
                Conv = kl.Conv2D if layer_type == 'C' else kl.Conv2DTranspose
            elif conv_d == 3:
                Conv = kl.Conv3D if layer_type == 'C' else kl.Conv3DTranspose

            return Conv(layer_specs['filters'],
                        kernel_size=layer_specs['kernel_size'],
                        strides=layer_specs.get('strides',default_stride),
                        padding=layer_specs.get('padding',"valid"),
                        data_format=layer_specs.get('data_format'),
                        activation=layer_specs.get('activation'))(x)
        elif layer_type == 'GN' or layer_type.lower() in ['group_norm','group_normalization']:
            return kl.GroupNormalization(groups=layer_specs.get('groups',32),
                                         axis=layer_specs.get('axis',-1),
                                         epsilon=layer_specs.get('epsilon',0.001),
                                         center=layer_specs.get('center',True),
                                         scale=layer_specs.get('scale',True))(x)
        elif layer_type == 'BN' or layer_type.lower() in ['batch_norm','batch_normalization']:
            return kl.BatchNormalization(axis=layer_specs.get('axis',-1),
                                         momentum=layer_specs.get('momentum',0.99),
                                         epsilon=layer_specs.get('epsilon',0.001),
                                         center=layer_specs.get('center',True),
                                         scale=layer_specs.get('scale',True))(x)
        elif layer_type == 'MP' or layer_type.lower() == 'max_pooling':
            return kl.MaxPooling2D(pool_size=layer_specs.get('pool_size',(2,2)),
                                   strides=layer_specs.get('strides'),
                                   padding=layer_specs.get('padding','valid'),
                                   data_format=layer_specs.get('data_format'))(x)
        elif layer_type == 'GAP' or layer_type.lower() in ['global_avg_pooling','global_average_pooling']:
            return kl.GlobalAveragePooling2D(data_format=layer_specs.get('data_format'))(x)
        elif layer_type == 'F' or layer_type.lower() in ['flat','flatten']:
            return kl.Flatten(data_format=layer_specs.get('data_format'))(x)
        elif layer_type == 'UP' or layer_type.lower() in ['upsampling','upsample','upsampling2d']:
            return kl.UpSampling2D(size=layer_specs.get('size',(2,2)),
                                   data_format=layer_specs.get('data_format'))(x)
        elif layer_type.lower() == 'custom':
            layer = layer_specs.get('layer')
            if layer is None:
                raise KeyError("No layer given")
            return layer(x)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    def _add_resnet_block(self,resnet_structs,ind,final_activation,x,allow_projection=True):
        '''
        Adds a ResNet block to the model with the given parameters
        
        :param resnet_structs (list or tuple): A list of structs of the form:
                                               [
                                               {'type':...,'specs':...},
                                               {'type':...,'specs':...},
                                               ...
                                               ]
        :param ind (int or str): The index of the resnet block
        :param final_activation (str or callable): The final activation function applied
                                                   after adding the residual
        :param x (KerasTensor): The tensor being passed through the model
        
        :return (KerasTensor): The tensor after the resnet block is applied
        '''
        pre_x = x
        out = x

        input_shape = keras.backend.int_shape(x)
        for sub_ind, struct in enumerate(resnet_structs):
            out = self._add_block(struct,f"{ind}.{sub_ind}",out)
        
        output_shape = keras.backend.int_shape(out)

        needs_projection = not shapes_equal(input_shape, output_shape)
        if needs_projection:
            if not allow_projection:
                raise ValueError(f"Residual block {ind}: input shape does not match output shape\n"
                                 f"input shape = {input_shape}, output shape = {output_shape}")
            pre_x = self._project(pre_x,output_shape)
        
        x = kl.Add()([pre_x, out])
        return kl.Activation(final_activation)(x)

    def _add_neural_block(self,layer_specs,ind,x):
        '''
        Adds a pretrained neural net into the model
        This is generally for transfer learning

        :param layer_specs (dict): A dictionary of the form
                                   {'model':...,'freeze':...}
                                   'model' is a keras.Model object
                                   'freeze' is a bool determining whether
                                   to freeze the weights and biases of
                                   the neural net
        :param ind (int or str): The index of this layer
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after being passed through
                               the given neural net
        '''
        model_in = layer_specs.get('model')

        # Checks whether the inputs were correct
        if model_in is None:
            raise KeyError(f"No model given for neural layer {ind}")
        if not isinstance(model_in, keras.Model):
            raise TypeError(f"Model given to layer {ind} must be a keras.Model object")
        
        freeze = layer_specs.get('freeze',False)

        # Cloning the model and retrieving the old weights and biases
        model = keras.models.clone_model(model_in)
        model.set_weights(model_in.get_weights())

        # Whether to freeze the weights and biases
        model.trainable = not freeze

        # Checks whether the output of the last layer/block is the expected shape
        expected = model.input_shape[1:]
        actual = keras.backend.int_shape(x)[1:]
        if expected != actual:
            raise ValueError(f"Neural net layer expects input shape: {expected}\n"
                             f"Instead got {actual}")

        return model(x)

    def _add_multioutput_block(self,layer_specs,ind,x):
        branches = layer_specs.get('branches')

        if branches is None:
            raise KeyError(f"Block {ind} must have branches")
        if not isinstance(branches,(list,tuple)) or len(branches) == 0:
            raise ValueError(f"Block {ind}: branches must be a non-empty list or tuple")
        
        outputs = []
        for branch_ind, branch in enumerate(branches):
            if not isinstance(branch, (list,tuple)) or len(branch) == 0:
                raise ValueError(f"Block {ind}, branch index {branch_ind}: "
                                 "each branch must be a non-empty list or tuple of layer specs")
            
            out = x
            for sub_ind, struct in enumerate(branch):
                out = self._add_block(struct,f"{ind}.{branch_ind}.{sub_ind}",out)
            
            outputs.append(out)
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs

    def _add_inception_block(self,inception_specs,ind,x):
        '''
        Adds an inception block to the model with the given parameters

        :param inception_specs (dict): A dictionary containing only one element with key 'branches'
                                       The associated value should be a non-empty list or tuple of
                                       the form:
                                       [[{'type':...},
                                         ...],
                                        [{'type':...},
                                         ...],
                                        ...
                                       ],
                                       indicating the different branches
        :param ind (int or str): The index of the inception block
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after the inception block is applied
        '''
        outputs = self._add_multioutput_block(inception_specs,ind,x)

        if not isinstance(outputs,list):
            return outputs
        
        shapes = [keras.backend.int_shape(out) for out in outputs]
        compare = shapes[0][1:-1]
        if any(s[1:-1] != compare for s in shapes[1:]):
            raise ValueError(f"Inception block {ind}: "
                             "all branch outputs need to have matching spatial dimensions\n"
                             f"Got shapes: {shapes}")
        
        return kl.Concatenate(axis=-1)(outputs)
    
    def _add_xception_block(self,xception_specs,ind,final_activation,x,allow_projection=True):
        '''
        Adds an xception block to the model with the given parameters

        :param xception_specs (list or tuple): A list of the form
                                               [{'filters':...,},{'filters':...},...]
        :param ind (int or str): The index of the xception block
        :param final_activation (str or callable): The final activation function applied
                                                   after adding the residual
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after the xception block is applied
        '''
        xception_specs = xception_specs.get('specs',xception_specs)
        
        pre_x = x
        out = x

        if len(x.shape) != 4:
            raise ValueError(f"Expected input to Conv2D to have rank 4, got shape {x.shape}")
        
        for sub_ind, spec in enumerate(xception_specs):
            if spec.get('filters') is None:
                raise KeyError(f"Filters must be given for layer {ind}.{sub_ind}")
            if spec.get('activation') is None:
                raise KeyError(f"No activation function given for layer {ind}.{sub_ind}")
            if spec.get('kernel_size') is None:
                raise KeyError(f"No kernel size given for convolutional layer {ind}.{sub_ind}")
            
            out = kl.SeparableConv2D(filters=spec['filters'],
                                     kernel_size=spec['kernel_size'],
                                     padding=spec.get('padding','same'),
                                     activation=spec.get('activation'))(out)
        
        input_shape = keras.backend.int_shape(pre_x)
        output_shape = keras.backend.int_shape(out)
        needs_projection = not shapes_equal(input_shape,output_shape)
        if needs_projection:
            if not allow_projection:
                raise ValueError(f"Xception block {ind}: input shape does not match output shape\n"
                                 f"input shape = {input_shape}, output shape = {output_shape}")
            pre_x = self._project(pre_x,keras.backend.int_shape(out))
        
        out = kl.Add()([pre_x,out])
        return kl.Activation(final_activation)(out)

    def _add_regressor_block(self,model,ind,x):
        '''
        Adds a layer to the model that is the output of a trained sklearn model
        WARNING: Backpropagation will stop at this layer

        :param model (scikit-learn regressor): A fully trained scikit-learn regressor
                                               (ex. GradientBoostingRegressor)
        :param ind (int or str): The index of this layer
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after being passed through the regressor
        '''
        if len(x.shape) != 2:
            raise ValueError("Expected input of model layer to have shape (n_samples, n_features)")

        try:
            return SKlearnLayer(model)(x)
        except Exception as e:
            raise RuntimeError(f"Exception found in regressor layer {ind}: {e}") from e

    def _add_block(self,struct,ind,x):
        '''
        Adds a block to the model with the given parameters
        Will use the previous _block methods

        :param struct (dict): A dictionary of the form {'type':...,'specs':...}
                              or {'type':...,'hyperparam1':...}
        :param ind (int or str): The index of the block
        :param x (KerasTensor): The tensor being passed through the model

        :return (KerasTensor): The tensor after the block is applied
        '''
        layer_type = struct['type'].replace(" ", "_")
        layer_specs = struct.get('specs',struct)

        if layer_type == 'R' or layer_type.lower() in ['resnet','residual']:
            block_structs = layer_specs.get('layers')

            return self._add_resnet_block(block_structs,ind,layer_specs.get("final_activation",'linear'),x,
                                          layer_specs.get('allow_projection',True))
        elif layer_type == 'I' or layer_type.lower() in ['inception','incep']:
            return self._add_inception_block(layer_specs,ind,x)
        elif layer_type == 'X' or layer_type.lower() in ['xcep','xception']:
            block_specs = layer_specs.get('xcep_specs')

            return self._add_xception_block(block_specs,ind,layer_specs.get('final_activation','linear'),x,
                                            layer_specs.get('allow_projection',True))
        elif layer_type.lower() == 'regressor':
            model = layer_specs.get('model')
            
            return self._add_regressor_block(model,ind,x)
        elif layer_type == 'NN' or layer_type.lower() == 'neural':
            return self._add_neural_block(layer_specs,ind,x)
        elif layer_type == 'multi-output':
            return self._add_multioutput_block(layer_specs,ind,x)
        else:
            return self._add_simple_block(layer_type,layer_specs,ind,x)


    ### SKLEARN METHODS ###

    def build_model(self):
        '''
        Builds the keras model from the given model structure

        :return (keras.Model): The fully built and compiled model
        '''
        self.is_multi_output_ = False

        input_shape = self.input_shape_

        inputs = kl.Input(shape=input_shape)

        x = inputs

        self._validate_hyperparams()

        structs = self.model_structure

        if self.build_setting == "quick":
            structs = parse_quick(structs)

        for ind,struct in enumerate(structs):
            if ind == len(structs) - 1:
                outputs = self._add_block(struct,ind,x)
            else:
                if struct['type'] == 'multi-output':
                    raise ValueError("Multi-output block must come last")
                
                x = self._add_block(struct,ind,x)
        
        self.is_multi_output_ = isinstance(outputs,list)

        if self.is_multi_output_:
            self.output_shape_ = [
                keras.backend.int_shape(output)[1:]
                for output in outputs
            ]
        else:
            self.output_shape_ = keras.backend.int_shape(outputs)[1:]

        # Creates and compiles the model
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=self._make_optimizer(),loss=self.loss,metrics=self.metrics)

        return model

    def fit(self, X, y, **fit_params):
        '''
        Trains the model on the given features and labels

        :param X (array-like): The features of shape (n_samples, ...)
        :param y (array-like): The labels of shape (n_samples, ...) or (n_samples,)
        :param fit_params: Any additional fit parameters used in Keras

        :return (self): The trained estimator
        '''
        if issparse(X):
            raise ValueError("Sparse input is not supported")
        
        X = np.asarray(X)

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        self.input_shape_ = self.input_shape if self.input_shape is not None else X.shape[1:]

        # Checks to see if the shapes are correct
        
        self.model_ = self.build_model()

        # If y is of shape (n_samples,), we need it to be of shape (n_samples,1)
        if self.is_multi_output_:
            self.y_was_1d_ = [
                np.asarray(target).ndim == 1
                for target in y
            ]

            y = [
                np.asarray(target).reshape(-1,1)
                if np.asarray(target).ndim == 1
                else np.asarray(target)
                for target in y
            ]
        else:
            y = np.asarray(y)

            self.y_was_1d_ = y.ndim == 1

            if y.ndim == 1:
                y = y.reshape(-1, 1)
        
        X,y = self._validate_data(X,y)

        callbacks = self._get_callbacks()

        if self.is_multi_output_:
            y = [target.astype(np.float32) for target in y]
        else:
            y = y.astype(np.float32)

        history = self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=self.shuffle,
            **fit_params,
        )

        # Stores the history, loss curve, and validation scores
        self.history_ = history.history
        self.loss_curve_ = history.history.get("loss")
        self.validation_scores_ = history.history.get("val_loss")

        return self

    def predict(self, X):
        '''
        Predicts the labels given the features

        :param X (array-like): The features of shape (n_samples, ...)

        :return (numpy.ndarray): The labels of shape (n_samples, ...) or (n_samples,)
        '''
        self._check_is_fitted()
        X = self._validate_data(X)

        pred = self.model_.predict(X, verbose=0)

        if not self.is_multi_output_:
            if self.y_was_1d_:
                return pred.ravel()
        else:
            return [
                p.ravel() if was_1d else p
                for p, was_1d in zip(pred, self.y_was_1d_)
            ]


        return pred