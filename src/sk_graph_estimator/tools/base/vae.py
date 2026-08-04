import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras.ops as ko
import keras.metrics as kmetrics

from .sae import SAE

def sampling(args):
    latent_mean, log_var = args
    eps = keras.random.normal(ko.shape(latent_mean))
    return latent_mean + ko.exp(0.5*log_var) * eps

class VAE(SAE):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         **kwargs)

        self.loss_metrics.update({
            'kl_loss': kmetrics.Mean()
        })
    
    def _get_loss(self,input):
        latent_mean, log_var, latent = self.encoder(input)
        output = self.decoder(latent)

        scale = np.prod(input.shape[1:])
        reco_loss = scale*ko.mean(ko.square(input - output))

        kl_loss = 1 + log_var - ko.square(latent_mean) - ko.exp(log_var)
        kl_loss = -0.5 * ko.sum(kl_loss, axis = -1)

        total_loss = reco_loss + kl_loss

        return total_loss, reco_loss, kl_loss
    
    def call(self, inputs):
        _, _, latent = self.encoder(inputs)
        output = self.decoder(latent)
        return output