import tensorflow as tf
from tensorflow.keras import models as km
from tensorflow.keras import metrics as kmet
from tensorflow.keras import ops as ko
import numpy as np

class PINN(km.Model):
    def __init__(self,X_data,mins,maxs,model,calc_eqn,calc_bound_eqn,**kwargs):
        super().__init__(*kwargs)
        self.X_data = X_data
        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)

        self.model = model
        self.calc_eqn = calc_eqn
        self.calc_bound_eqn = calc_bound_eqn

        self.loss_tracker = kmet.Mean()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _get_loss(self,X_r):
        loss = ko.sum(ko.square(self.calc_eqn(X_r)))

        for i in range(len(self.X_data)):
            loss += ko.sum(ko.square(self.calc_bound_eqn(self.X_data,i)))

        return loss

    def train_step(self,X_r):
        with tf.GradientTape(persistent=True) as tape:
            trainable_variables = self.trainable_variables

            loss = self._get_loss(X_r)

        grad_theta = tape.gradient(loss,trainable_variables)

        self.optimizer.apply_gradients(zip(grad_theta,trainable_variables))

        self.loss_tracker.update_state(loss)

        return {'loss':self.loss_tracker.result()}

    def test_step(self,X_r):
        loss = self._get_loss(X_r)
        self.loss_tracker.update_state(loss)

        return {'loss':self.loss_tracker.result()}

    def call(self,inputs):
        return self.model(inputs)

