'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout, Activation
from keras.layers import Convolution2D, Deconvolution2D
from keras.layers import advanced_activations as aa
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization as BN

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
nb_filters = 64
# convolution kernel size
nb_conv = 3

latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0

batch_size = 100
nb_epoch = 5
lr = 0.004

################################################################

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

if K.image_dim_ordering() == 'th':
    original_img_size = (img_chns, img_rows, img_cols)
    output_shape1 = (batch_size, nb_filters, 14, 14)
    output_shape2 = (batch_size, nb_filters, 29, 29)
else:
    original_img_size = (img_rows, img_cols, img_chns)
    output_shape1 = (batch_size, 14, 14, nb_filters)
    output_shape2 = (batch_size, 29, 29, nb_filters)

################################################################

# Note: it seems like we cannot use multiple input layers.
#       Always use the same instance of x.

x = Input(shape=original_img_size)

encoder_pre = Sequential([
    Convolution2D(img_chns, 2, 2,
                  border_mode='same', activation='relu',
                  input_shape=original_img_size),
    # Convolution2D(img_chns, 2, 2,
    #               border_mode='same', bias=False,
    #               input_shape=original_img_size),
    # Activation('relu'),
    Convolution2D(nb_filters, 2, 2,
                  border_mode='same', activation='relu',
                  subsample=(2, 2)),
    # Dropout(0.2),
    Convolution2D(nb_filters, nb_conv, nb_conv,
                  border_mode='same', activation='relu',
                  subsample=(1, 1)),
    Convolution2D(nb_filters, nb_conv, nb_conv,
                  border_mode='same', activation='relu',
                  subsample=(1, 1)),
    Flatten(),
    Dense(intermediate_dim*5, activation='relu'),
    Dropout(0.2),
])
encoder_pre.summary()

h = encoder_pre(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_var])

# variational encoder
v_encoder = Model(input=x,output=z)

################################################################

decoder = Sequential([
    Dense(intermediate_dim, activation='relu',
          input_shape=(latent_dim,)),
    Dense(nb_filters * 14 * 14, activation='relu'),
    Reshape(output_shape1[1:]),
    Deconvolution2D(nb_filters, nb_conv, nb_conv,
                    output_shape1,
                    border_mode='same',
                    subsample=(1, 1),
                    activation='relu'),
    Deconvolution2D(nb_filters, nb_conv, nb_conv,
                    output_shape1,
                    border_mode='same',
                    subsample=(1, 1),
                    activation='relu'),
    Deconvolution2D(nb_filters, 2, 2,
                    output_shape2,
                    border_mode='valid',
                    subsample=(2, 2),
                    activation='relu'),
    Convolution2D(img_chns, 2, 2, border_mode='valid', activation='sigmoid')
])

################################################################

y2 = decoder(v_encoder(x))
vae = Model(input=x,output=y2)
vae.compile(optimizer='rmsprop', loss=vae_loss, lr=lr)

################################################################
# training

# train the VAE on MNIST digits
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)

from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='vae-conv-deconv'),
                   CSVLogger("vae-conv-deconv/loss.csv"),
                   # EarlyStopping(patience=6,verbose=1,mode='min'),
                   ReduceLROnPlateau(verbose=1,patience=20,factor=0.5,mode='min',epsilon=0.0001)
        ])

vae.save('vae-conv-deconv.h5')

################################################################
# plotting

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.savefig("latent.png")

# build a digit generator that can sample from the learned distribution
generator = decoder

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = generator.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.savefig("manifold.png")

