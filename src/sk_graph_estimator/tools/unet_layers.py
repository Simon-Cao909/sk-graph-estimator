import tensorflow.keras.layers as kl
import tensorflow.keras.saving as ks

@ks.register_keras_serializable()
class Conv3Layer(kl.Layer):
    def __init__(self,n_chan,kernel_size,n_groups,**kwargs):
        super().__init__(**kwargs)
        self.num_chan = n_chan
        self.Conv2D_1 = kl.Conv2D(n_chan,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='linear')
        self.Norm_1 = kl.GroupNormalization(groups=n_groups)
        self.Relu_1 = kl.RelU()

        self.Conv2D_2 = kl.Conv2D(n_chan,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='linear')
        self.Norm_2 = kl.GroupNormalization(groups=n_groups)
        self.Relu_2 = kl.ReLU()

    def call(self,inputs):
        x = self.Conv2D_1(inputs)
        x = self.Norm_1(x)
        x = self.Relu_1(x)

        x = self.Conv2D_2(x)
        x = self.Norm_2(x)
        x = self.Relu_2(x)

        return x
    
@ks.register_keras_serializable()
class UnetDownLayer(kl.Layer):
    def __init__(self,n_chan,kernel_size,n_groups,pool_size,**kwargs):
        super().__init__(**kwargs)
        self.num_chan = n_chan
        self.Conv = Conv3Layer(n_chan,
                               kernel_size=kernel_size,
                               n_groups=n_groups)
        self.Pool = kl.MaxPool2D(pool_size=pool_size,
                                 strides = 2)
    
    def call(self,inputs):
        x = self.Conv(inputs)
        skip = x
        x = self.Pool(x)
        return x,skip

@ks.register_keras_serializable()
class UnetUpLayer(kl.Layer):
    def __init__(self,n_chan,kernel_size,n_groups,**kwargs):
        super().__init__(**kwargs)
        self.num_chan = n_chan
        self.Conv2DTranspose = kl.Conv2DTranspose(n_chan,
                                                  kernel_size=(2,2),
                                                  strides=(2,2),
                                                  padding='same',
                                                  activation='linear')
        self.Conv = Conv3Layer(n_chan,kernel_size,n_groups)
        self.Concat = kl.Concatenate(axis=-1)
    
    def call(self,inputs):
        inp, skip = inputs

        x = self.Conv2DTranspose(inp)
        x = self.Concat([x,skip])

        x = self.Conv(x)

        return x

@ks.register_keras_serializable()
class UnetBottleneck(kl.Layer):
    def __init__(self,n_chan,kernel_size,n_groups,**kwargs):
        super().__init__(**kwargs)

        self.Conv1 = Conv3Layer(n_chan,kernel_size,n_groups)
        self.Conv2 = Conv3Layer(n_chan,kernel_size,n_groups)

    def call(self,x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        return x