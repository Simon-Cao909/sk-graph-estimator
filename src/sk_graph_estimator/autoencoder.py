from tensorflow import keras
from tensorflow.keras import layers as kl
import numpy as np

from .estimator import SKGraphEstimator
from .tools.quick_build_parser import parse_quick
from .tools.sae import SAE
from .tools.vae import VAE, sampling

class SKGraphAutoencoder(SKGraphEstimator):
    '''
    SKGraphAutoencoder is a subclass of SKGraphEstimator that is meant to
    create standard and variational autoencoders
    '''
    def __init__(
        self,
        encoder_structure,
        decoder_structure,
        model_type='standard',
        build_setting="normal",
        input_shape=None,
        epochs=100,
        batch_size=32,
        early_stopping=True,
        n_iter_no_change=10,
        validation_split=0.1,
        verbose=1,
        optimizer="adam",
        learning_rate=1e-3,
        random_state=None,
        shuffle=True,
    ):
        '''
        Attributes
        - encoder_structure (list or tuple): model structure for the encoder
                                             See architecture.md for how to format this
        - decoder_structure (list or tuple): model structure for the decoder
                                             See architecture.md for how to format this
        - model_type (str, default='standard'): Specifies the autoencoder type
                                                Must be either 'standard' or 'variational'
        - build_setting (str, default='normal'): Decides the format of model_structure
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
        - optimizer (str, default='adam'): The optimizer used in training. See Keras for possibilities
        - learning_rate (float, default=1e-4): The learning rate for training
        - random_state (int or None, default=None): The random state. Used for reproducible results
        - shuffle (bool, default=True): Whether to shuffle the data before training
        '''
        model_structure = list(encoder_structure) + list(decoder_structure)

        super().__init__(model_structure=model_structure,
                         build_setting=build_setting,
                         input_shape=input_shape,
                         epochs=epochs,
                         batch_size=batch_size,
                         early_stopping=early_stopping,
                         n_iter_no_change=n_iter_no_change,
                         validation_split=validation_split,
                         verbose=verbose,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         random_state=random_state,
                         shuffle=shuffle)

        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure

        self.model_type = model_type


    ### HELPER FUNCTIONS ###

    def _build_encoder(self):
        '''
        Builds the encoder half of the autoencoder using
        the given encoder_structure

        :return (keras.Model): The encoder model
        '''
        model_type = self.model_type.lower()
        input_shape = self.input_shape_

        encoder_inputs = kl.Input(shape=input_shape)

        x = encoder_inputs

        encoder_structs = self.encoder_structure

        if self.build_setting == 'quick':
            encoder_structs = parse_quick(encoder_structs)

        for ind,struct in enumerate(encoder_structs):
            if struct['type'] == 'multi-output':
                raise NotImplementedError("Multi-output encoders are not supported")

            if ind == len(encoder_structs) - 1:
                if model_type == 'standard':
                    latent = self._add_block(struct,ind,x)
                    encoder_outputs = latent
                elif model_type == 'variational':
                    latent_mean = self._add_block(struct,ind,x)
                    log_var = self._add_block(struct,ind,x)
                    latent = kl.Lambda(sampling)([latent_mean,log_var])
                    encoder_outputs = [latent_mean, log_var, latent]
            else:
                x = self._add_block(struct,ind,x)
        
        self.latent_shape_ = keras.backend.int_shape(latent)[1:]
        self.encoder_ = keras.Model(inputs=encoder_inputs,outputs=encoder_outputs)
    
    def _build_decoder(self):
        '''
        Builds the decoder half of the model structure using
        the given decoder_structure

        :return (keras.Model): The decoder model
        '''
        decoder_inputs = keras.Input(shape=self.latent_shape_)

        x = decoder_inputs

        decoder_structs = self.decoder_structure

        if self.build_setting == 'quick':
            decoder_structs = parse_quick(decoder_structs)

        for ind, struct in enumerate(decoder_structs):
            if struct['type'] == 'multi-output':
                raise NotImplementedError("Multi-output decoders are not supported")
            
            if ind == len(decoder_structs) - 1:
                decoded = self._add_block(struct,ind,x)
            else:
                x = self._add_block(struct,ind,x)
        
        self.output_shape_ = keras.backend.int_shape(decoded)[1:]
        self.decoder_ = keras.Model(inputs=decoder_inputs,outputs=decoded)

    def _use_model(self,arr,which):
        '''
        Predicts using the encoder or decoder

        :param arr (array-like): The input array for the encoder
                                 or latent array for the decoder
        :param which (str): The model to be used
                            Must be either 'encoder' or 'decoder'
        
        :return (np.ndarray): The output of the model
        '''
        self._check_is_fitted()
        arr = np.asarray(arr)

        expec_shape = self.input_shape_ if which == 'encoder' else self.latent_shape_
        model = self.model_.encoder if which == 'encoder' else self.model_.decoder

        one_sample = arr.shape == expec_shape

        if one_sample:
            arr = arr[None]

        if arr.shape[1:] != expec_shape:
            raise ValueError(
                f"The features have {arr.shape[1:]} shape, but this model was fitted with "
                f"{self.input_shape_} input shape."
            )
        
        output = model.predict(arr)

        if which == 'encoder':
            output = output[2]
        
        return output[0] if one_sample else output


    ### SKLEARN METHODS ###

    def build_model(self):
        '''
        Builds and compiles the autoencoder using the given
        encoder_structure and decoder_structure

        :return (keras.Model): The autoencoder model
        '''
        self.is_multi_output_ = False

        self._validate_hyperparams()
        model_type = self.model_type.lower()

        self._build_encoder()
        self._build_decoder()

        if model_type == 'standard':
            AutoEncoder = SAE
        elif model_type == 'variational':
            AutoEncoder = VAE
        else:
            raise ValueError("Model type must be either "
                             "'standard' or 'variational'")
        
        model = AutoEncoder(self.encoder_,self.decoder_)
        model.compile(optimizer=self._make_optimizer())

        return model
    
    def fit(self, X, y = None, **fit_params):
        '''
        Trains the model on the given features

        :param X (array-like): The features of shape (n_samples, ...)
        :param y (None): Leave this as None
        :param fit_params: Any additional fit parameters used in Keras

        :return (self): The trained autoencoder
        '''
        X = np.asarray(X)

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        self.input_shape_ = self.input_shape if self.input_shape is not None else X.shape[1:]
        
        self.model_ = self.build_model()

        X = self._validate_data(X)

        if self.output_shape_ != self.input_shape_:
            raise ValueError(
                "Input shape and output shape must match!\n"
                f"Current input shape: {self.input_shape_}\n"
                f"Current output shape: {self.output_shape_}"
            )

        callbacks = self._get_callbacks()

        history = self.model_.fit(
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=self.shuffle,
            **fit_params,
        )
    
        self.history_ = history.history
        self.loss_curve_ = history.history.get("loss")
        self.validation_scores_ = history.history.get("val_loss")

        return self
        
    def score(self, X, y=None):
        '''
        Scores the model based on how it performs on given data

        :param X (array-like): The features of shape (n_samples, ...)
        :param y (None): Leave this as None

        :return (float): Negative MSE between X and the prediction
        '''
        pred = self.predict(X)
        return -np.mean((X - pred)**2)
    
    
    ### AUTOENCODER SPECIFIC METHODS ###

    def encode(self,X):
        '''
        Encodes the given input

        :param X (array-like): The input array of shape (n_samples, *input_shape_)
                               or input_shape_

        :return (np.ndarray): The latent representation of X
                              of shape (n_samples, *latent_shape_)
                              or latent_shape_
        '''
        return self._use_model(X,'encoder')
    
    def decode(self,latent):
        '''
        Decodes the given latent representation

        :param latent (array-like): The latent array of shape (n_samples, *latent_shape_)
                                    or latent_shape_

        :return (np.ndarray): The output of the decoder
                              of shape (n_samples, *output_shape_)
                              or output_shape_
        '''
        return self._use_model(latent,'decoder')