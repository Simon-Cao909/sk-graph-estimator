from tensorflow import keras
from tensorflow.keras import layers as kl
import numpy as np

from .estimator import SKGraphEstimator
from .tools.quick_build_parser import parse_quick
from .tools.sae import SAE
from .tools.vae import VAE, sampling

class SKGraphAutoencoder(SKGraphEstimator):
    def __init__(self,encoder_structure,decoder_structure,model_type='standard',**kwargs):
        model_structure = list(encoder_structure) + list(decoder_structure)

        super().__init__(model_structure=model_structure,**kwargs)

        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure

        self.model_type = model_type

    def build_encoder(self):
        model_type = self.model_type.lower()
        input_shape = self.input_shape_

        encoder_inputs = kl.Input(shape=input_shape)

        x = encoder_inputs

        encoder_structs = self.encoder_structure

        if self.build_setting == 'quick':
            encoder_structs = parse_quick(encoder_structs)

        for ind,struct in enumerate(encoder_structs):
            if ind == len(encoder_structs) - 1:
                if model_type == 'standard':
                    latent = self._add_block(struct,ind,x)
                    encoder_outputs = latent
                elif model_type == 'variational':
                    latent_mean = self._add_block(struct,ind,x)
                    latent_log = self._add_block(struct,ind,x)
                    latent = kl.Lambda(sampling)([latent_mean,latent_log])
                    encoder_outputs = [latent_mean, latent_log, latent]
            else:
                x = self._add_block(struct,ind,x)
        
        self.latent_shape_ = keras.backend.int_shape(latent)[1:]
        self.encoder_ = keras.Model(inputs=encoder_inputs,outputs=encoder_outputs)
    
    def build_decoder(self):
        decoder_inputs = keras.Input(shape=self.latent_shape_)

        x = decoder_inputs

        decoder_structs = self.decoder_structure

        for ind, struct in enumerate(decoder_structs):
            if ind == len(decoder_structs) - 1:
                decoded = self._add_block(struct,ind,x)
            else:
                x = self._add_block(struct,ind,x)
        
        self.output_shape_ = keras.backend.int_shape(decoded)[1:]
        self.decoder_ = keras.Model(inputs=decoder_inputs,outputs=decoded)

    def build_model(self):
        self._validate_hyperparams()
        model_type = self.model_type.lower()

        self.build_encoder()
        self.build_decoder()

        if model_type == 'standard':
            AutoEncoder = SAE
        elif model_type == 'variational':
            AutoEncoder = VAE
        
        model = AutoEncoder(self.encoder_,self.decoder_)
        model.compile(optimizer=self._make_optimizer())

        return model
    
    def fit(self, X, y = None, **fit_params):
        X = np.array(X)

        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)

        self.input_shape_ = self.input_shape if self.input_shape is not None else X.shape[1:]

        if self.input_shape_ != X.shape[1:]:
            raise ValueError(
                f"input_shape={self.input_shape_}, but features have shape {X.shape[1:]}"
            )
        
        self.model_ = self.build_model()

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
        pred = self.predict(X)
        return -np.mean((X - pred)**2)