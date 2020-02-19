import gzip
import pickle
import sys, os
import numpy as np

f = gzip.open('mnist.pkl.gz', 'rb')

if sys.version_info < (3,):
    data = pickle.load(f)
else:
    data = pickle.load(f, encoding='bytes')
f.close()

(x_train, y_train), (x_test, y_test) = data

assert(len(x_test) == len(y_test))
assert(len(x_train) == len(y_train))

print("No of training examples: ", len(x_train))

print("shape of training examples: ", x_train.shape)

N, W, H = x_train.shape
D = W * H
x_test = x_test.reshape(-1, D)
x_train = x_train.reshape(-1, D)

print("shape of training examples: ", x_train.shape)

# not sure about this
latent_dim = 100

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim):
    i = Input(shape=(latent_dim,))
    x = Dense(256, activation=LeakyReLU(alpha=0.3))(i)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.3))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.3))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(D, activation='tanh')(x)
    
    model = Model(i, x)
    
    return model

def build_discriminator(img_size):
    i = Input(shape=(img_size,))
    x = Dense(1025, activation=LeakyReLU(alpha=0.3))(i)
    x = Dense(512, activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(256, activation=LeakyReLU(alpha=0.3))(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(i, x)
    
    return model

def create_gan():
    discriminator_model = build_discriminator(D)
    generator_model = build_generator(latent_dim)

    discriminator_model.compile(loss='binary_crossentropy', 
                      optimizers=Adam(0.002, 0.5),
                      metrics = ['accuracy'])

    z = Input(shape=(latent_dim,))
    fake_img = generator_model(z)
    discriminator_model.trainable = False
    fake_pred = discriminator_model(fake_img)
    combine_model = Model(z, fake_pred)

    return combine_model

gan = create_gan()

gan.compile(loss='binary_crossentropy',
                      optimizers=Adam(0.002, 0.5),
                      metrics = ['accuracy'])

d_loss = []
g_loss = []

epochs = 20000
batch_size = 32
sample_period = 2000

ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')
