import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History 

from keras.utils import plot_model
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects

def sampling(args):
    
    epsilon_std = 1.0
    
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch, dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

class Autoencoder:

    def __init__(self, x_sz, y_sz=1, z_dim= 64, variational = False, name=[]):
        self.name = name
        self.x_sz = x_sz
        self.y_sz = y_sz
        self.m2m = []
        self.m2zm = []
        self.zm2m = []
        self.variational = variational
        self.z_dim = z_dim

    def encoder2D(self):
        #define the simple autoencoder
        input_image = Input(shape=(self.x_sz,)) 

        #image encoder
        _ = Dense(512)(input_image)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(256)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(128)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(64)(_)

        if not self.variational:
            encoded_image = Dense(self.z_dim)(_)
        else:
            _ = Dense(self.z_dim)(_)
            z_mean_m = Dense(self.z_dim)(_)
            z_log_var_m = Dense(self.z_dim)(_)
            encoded_image = Lambda(sampling)([z_mean_m, z_log_var_m])
            return input_image, encoded_image, z_mean_m, z_log_var_m

        return input_image, encoded_image

    def decoder2D(self, encoded_image):
        #image decoder
        _ = Dense(64)(encoded_image)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(128)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(256)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        _ = Dense(512)(_)
        _ = LeakyReLU(alpha=0.3)(_)

        decoded_image = Dense(self.x_sz)(_)

        return decoded_image

    def train_autoencoder2D(self, x_train, load = False):
        #set loss function, optimizer and compile
        input_image, encoded_image = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)

        self.m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.m2m.compile(optimizer=opt, 
                        loss="mse", 
                        metrics=['mse'])

        #get summary of architecture parameters and plot arch. diagram
        self.m2m.summary()
        plot_model(self.m2m, to_file='AE_m2m.png')

        #train the neural network
        if not load:
            plot_losses = util.PlotLosses()
            self.m2m.fit(x_train, x_train,        
                            epochs=100,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.3,
                            callbacks=[plot_losses])
            #save trained model
            self.m2m.save('AE_m2m.h5')
        else:
            #load an already trained model
            print("Trained model loaded")
            self.m2m = load_model('AE_m2m.h5')

        #set the encoder model
        input_image_f = Input(shape=(self.x_sz,)) 
        _ = self.m2m.layers[1](input_image_f)
        for i in range(2, 8):
            _ = self.m2m.layers[i](_)
        encoded_image_f = self.m2m.layers[8](_)
        self.m2zm = Model(input_image_f, encoded_image_f)

        #set the decoder model
        zm_dec = Input(shape=(self.z_dim, )) 
        _ = self.m2m.layers[9](zm_dec)
        for i in range(10, 17):
            _ = self.m2m.layers[i](_)
        decoded_image_ = self.m2m.layers[17](_)
        self.zm2m = Model(zm_dec, decoded_image_)

    def train_var_autoencoder2D(self, x_train, load = False):
        #set loss function, optimizer and compile
        input_image, encoded_image, z_mean, z_log_var = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)

        #define the variational loss and mse loss (equal weighting)
        def vae_loss(input_image, decoded_image):
            recons_loss = K.sum(mse(input_image, decoded_image))                
            kl_loss = (- 0.5) * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(recons_loss + 0.1*kl_loss)

        #add custom loss 
        get_custom_objects().update({"vae_loss": vae_loss})

        self.m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-6)
        self.m2m.compile(optimizer=opt, 
                        loss=vae_loss)

        #get summary of architecture parameters and plot arch. diagram
        self.m2m.summary()
        plot_model(self.m2m, to_file='AE_m2m_var.png')

        #train the neural network
        if not load:
            plot_losses = util.PlotLosses()
            self.m2m.fit(x_train, x_train,        
                            epochs=10,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.3,
                            callbacks=[plot_losses])
            #save trained model
            self.m2m.save('AE_m2m_var.h5')
        else:
            #load an already trained model
            print("Trained model loaded")
            self.m2m = load_model('AE_m2m_var.h5')

        #set the encoder model
        self.m2zm = Model(input_image, encoded_image)

        #set the decoder model
        zm_dec = Input(shape=(self.z_dim, )) 
        _ = self.m2m.layers[12](zm_dec)
        for i in range(13, 20):
            _ = self.m2m.layers[i](_)
        decoded_image_ = self.m2m.layers[20](_)
        self.zm2m = Model(zm_dec, decoded_image_)


