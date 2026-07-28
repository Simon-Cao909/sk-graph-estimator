import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.ops as ko
import keras.metrics as kmetrics

class SAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        
        self.encoder = encoder
        self.decoder = decoder

        self.loss_metrics = {
            'loss': kmetrics.Mean(),
            'reco_loss': kmetrics.Mean(),
        }
    
    def _get_loss(self,input):
        output = self(input)
        scale = np.prod(input.shape[1:])
        reco_loss = scale*ko.mean(ko.square(input - output))

        return reco_loss,reco_loss
    
    def _update_loss(self,losses):
        for ind, key in enumerate(self.loss_metrics.keys()):
            self.loss_metrics[key].update_state(losses[ind])
    
    @property
    def metrics(self):
        return list(self.loss_metrics.values())

    def call(self, inputs):
        latent = self.encoder(inputs)
        output = self.decoder(latent)
        return output
    
    def train_step(self,input):
        with tf.GradientTape() as tape:
            losses = self._get_loss(input)
        
        tot_loss = losses[0]

        gradient = tape.gradient(tot_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_weights))

        self._update_loss(losses)

        return {key:val.result() for key,val in self.loss_metrics.items()}

    def test_step(self,input):
        losses = self._get_loss(input)
        self._update_loss(losses)

        return {key:val.result() for key,val in self.loss_metrics.items()}