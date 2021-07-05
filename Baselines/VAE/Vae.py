# coding=utf-8
from datetime import datetime

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat,savemat
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Lambda,Dropout,Reshape, UpSampling1D,Activation, Conv2D, Conv1DTranspose,MaxPooling2D, Flatten, LSTM,Conv1D,MaxPooling1D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import mse
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,roc_auc_score,plot_roc_curve,confusion_matrix
from tensorflow.keras import backend as K
def plot_series():
    ir = 100
    synthistic = np.loadtxt(
        open('/usr/CSMOTE/Datasets/synthistic/251/vae_4_'+str(ir)+'.csv'), delimiter=",",
        skiprows=0)
    for i in range(20):
        plt.figure()
        plt.plot(range(1600), synthistic[i].reshape((1600)), linewidth=1.0)
        plt.xlabel('value')
        plt.ylabel('time')
        plt.legend()
        plt.show()
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def plot_loss(history,modal,batch_size):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    min_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(' ml:' + str("{:.3}".format(min_loss)) + ' mvl:' + str(
        "{:.3}".format(min_val_loss))+ ' ir:' + ' batch_size:' + str(batch_size))
    plt.show()
def get_vae(latent_dim):
    input = Input(shape=(577, 1))
    conv = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(input)
    conv = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Dropout(0.5)(conv)
    shape = K.int_shape(conv)
    print(shape)
    flatten = Flatten()(conv)
    z_mean = Dense(latent_dim, name='z_mean')(flatten)
    z_log_var = Dense(latent_dim, name='z_log_var')(flatten)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    reshape = Reshape((shape[1], shape[2]))(Dense(shape[1] * shape[2], activation='relu')(latent_inputs))
    cont = Conv1DTranspose(filters=16, kernel_size=3, activation='relu', padding='same')(reshape)
    output = Conv1DTranspose(filters=1, kernel_size=3, activation='relu', padding='same')(cont)
    decoder = Model(latent_inputs, output, name='decoder')
    decoder.summary()
    output = decoder(encoder(input)[2])
    vae = Model(input, output, name='vae')
    vae.summary()
    reconstruction_loss = mse(input, output)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    return vae,encoder,decoder,reconstruction_loss,kl_loss
def fit_vae(train_x,test_x,learning_ratio,latent_dim,kl_coefficient,adam,epochs,batch_size):
    vae, encoder, decoder, reconstruction_loss, kl_loss =get_vae(latent_dim)
    vae_loss = K.mean(reconstruction_loss) + K.mean(kl_coefficient * kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=adam)
    history = vae.fit(train_x, epochs=epochs, batch_size=batch_size, validation_data=(test_x, None), verbose=1)
    plot_loss(history,'vae',batch_size)
    models = vae, encoder, decoder
    plot_hidden_result(models, train_x, kl_coefficient)
    plot_recons_result(models, train_x, kl_coefficient)
    plot_hidden_result(models, test_x, kl_coefficient)
    plot_recons_result(models, test_x, kl_coefficient)
    vae.save("/usr/CSMOTE/Vae/Car/1vae_" + str(batch_size) + "_" + str(learning_ratio) + "_" + str(latent_dim)+ ".h5")
    encoder.save("/usr/CSMOTE/Vae/Car/1encoder_" + str(batch_size) + "_" + str(learning_ratio) + "_" + str(latent_dim) + ".h5")
    decoder.save("/usr/CSMOTE/Vae/Car/1decoder_" + str(batch_size) + "_" + str(learning_ratio) + "_" + str(latent_dim) + ".h5")
def generate():
    h = 2
    n = 49
    z_sample = np.random.normal(0, 1, (n, h))
    decoder = load_model('/usr/CSMOTE/Vae/Car/decoder_4_0.0005_'+str(h)+'.h5')
    decoder.summary()
    sample = decoder.predict(z_sample)
    print(sample.shape)
    np.savetxt('/usr/CSMOTE/Datasets/synthistic/Car/vae_' + str(h)  + '.csv', sample.reshape((n, 577)),
               delimiter=',')
def plot_hidden_result(models,data,kl_coefficient):
    vae,encoder,decoder = models
    z_mean, z_log_var, z = encoder.predict(data, batch_size=1)
    z_mean = np.array(z_mean)
    plt.figure(figsize=(12, 10))
    plt.scatter(z[:, 0], z[:, 1],s=20)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axhline(y=0, c='gray')
    plt.axvline(x=0, c='gray')
    plt.title(f'KL coefficient = {kl_coefficient}', fontdict={'fontsize': 'xx-large'})
    plt.show()
def plot_recons_result(models,data,kl_coefficient):
    vae,encoder,decoder = models
    data_pre = vae.predict(data, batch_size=1)
    data_pre = data_pre.reshape(data_pre.shape[0] * data_pre.shape[1])
    data= data.reshape(data.shape[0] * data.shape[1])
    plt.figure()
    plt.title(f'KL coefficient = {kl_coefficient}', fontdict={'fontsize': 'xx-large'})
    plt.plot(range(577), data[0:577].tolist())
    plt.plot(range(577), data_pre[0:577].tolist())
    plt.show()
def process_vae():
    train_min = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/train_min.csv'),
        delimiter=",",
        skiprows=0)
    test_min = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/test_min.csv'),
        delimiter=",",
        skiprows=0)
