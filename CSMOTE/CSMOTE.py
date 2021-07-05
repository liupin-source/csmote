import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 这一行注释掉就是使用cpu，不注释就是使用gpu
# from pandas import  Series

import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, concatenate,Dropout,Lambda,Dense,Reshape,Conv1D,Conv1DTranspose,Flatten,MaxPooling1D,UpSampling1D
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score,precision_score,recall_score
from tensorflow.keras.models import load_model


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def plot_loss(history,kl_coefficient):
    plt.figure()
    plt.plot(history.history['loss'], label='tra_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    min_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(' ml:' + str("{:.3}".format(min_loss)) + ' mvl:' + str(
            "{:.3}".format(min_val_loss))+' kl_coefficient:'+str(kl_coefficient))

    return min_loss,min_val_loss
def get_csmote(latent_dim,input_dim):

    input_shape = (input_dim,1)
    tg_inputs = Input(shape=input_shape, name='tg_input')
    bg_inputs = Input(shape=input_shape, name='bg_input')
    z_conv1 = Conv1D(filters=32, kernel_size=3,activation='relu', padding='same')
    z_conv2 = Conv1D(filters=16, kernel_size=3,activation='relu', padding='same')
    z_conv3 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')
    z_h_flatten =Flatten()

    z_mean_layer = Dense(latent_dim, name='z_mean')
    z_log_var_layer = Dense(latent_dim, name='z_log_var')
    z_layer = Lambda(sampling, output_shape=(latent_dim,), name='z')
    def z_encoder(inputs):
        z_h = z_conv1(inputs)
        z_h = z_conv2(z_h)
        z_h = z_conv3(z_h)
        shape = K.int_shape(z_h)
        z_h = z_h_flatten(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z,shape

    s_conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    s_conv2 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')
    s_conv3 = Conv1D(filters=8, kernel_size=3,activation='relu', padding='same')
    s_h_flatten = Flatten()
    s_mean_layer = Dense(latent_dim, name='s_mean')
    s_log_var_layer = Dense(latent_dim, name='s_log_var')
    s_layer = Lambda(sampling, output_shape=(latent_dim,), name='s')
    def s_encoder(inputs):
        s_h = s_conv1(inputs)
        s_h = s_conv2(s_h)
        s_h = s_conv3(s_h)
        s_h = s_h_flatten(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    tg_z_mean, tg_z_log_var, tg_z,shape = z_encoder(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder(bg_inputs)
    z_encoder = Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')
    csmote_latent_inputs = Input(shape=(2 * latent_dim,), name='sampled')

    csmote_h = Dense(shape[1] * shape[2], activation='relu')(cvae_latent_inputs)
    csmote_h = Reshape((shape[1], shape[2]))(cvae_h)
    cont2 = Conv1DTranspose(filters=8, kernel_size=3, activation='relu', padding='same')(cvae_h)
    cont3 = Conv1DTranspose(filters=16, kernel_size=3, activation='relu', padding='same')(cont2)
    output = Conv1DTranspose(filters=1, kernel_size=3, activation='relu', padding='same')(cont3)
    csmote_decoder = Model(inputs=cvae_latent_inputs, outputs=output, name='decoder')
    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(concatenate([tg_z, tg_s], -1))
    zeros = Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(concatenate([tg_z, zeros], -1))

    csmote = Model(inputs=[tg_inputs, bg_inputs],outputs=[tg_outputs, bg_outputs],name='contrastive_vae')
    csmote.summary()

    csmote_fg = Model(inputs=tg_inputs,outputs=fg_outputs,name='contrastive_vae_fg')
    tg_rec_loss = mse(tg_inputs, tg_outputs)
    bg_rec_loss = mse(bg_inputs, bg_outputs)
    tg_z_kl_loss = 1 + tg_z_log_var - K.square(tg_z_mean) - K.exp(tg_z_log_var)
    tg_s_kl_loss = 1 + tg_s_log_var - K.square(tg_s_mean) - K.exp(tg_s_log_var)
    bg_s_kl_loss = 1 + bg_s_log_var - K.square(bg_s_mean) - K.exp(bg_s_log_var)
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder,tg_rec_loss,bg_rec_loss,tg_z_kl_loss,tg_s_kl_loss,bg_s_kl_loss
def plot_csmote_result(models,data,kl_coefficient,batch_size):
    csmote,z_encoder, s_encoder,decoder = models
    fault_train_set, normal_train_set=data

    # tg z_encoder
    tg_z_mean, tg_z_log_var, tg_z = z_encoder.predict(fault_train_set, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(tg_z_mean[:, 0], tg_z_mean[:, 1],s=20)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axhline(y=0, c='gray')
    plt.axvline(x=0, c='gray')
    plt.title(f'tg_z_encoder KL coefficient = {kl_coefficient}', fontdict={'fontsize': 'xx-large'})
    plt.show()
    # tg s_encoder
    tg_s_mean, tg_s_log_var, tg_s = s_encoder.predict(fault_train_set, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(tg_s_mean[:, 0], tg_s_mean[:, 1],s=20)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axhline(y=0, c='gray')
    plt.axvline(x=0, c='gray')
    plt.title(f'tg_s_encoder KL coefficient = {kl_coefficient}', fontdict={'fontsize': 'xx-large'})
    plt.show()
    # bg s_encoder
    bg_s_mean, bg_s_log_var, bg_s = s_encoder.predict(normal_train_set, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(bg_s_mean[:, 0], bg_s_mean[:, 1],s=20)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.axhline(y=0, c='gray')
    plt.axvline(x=0, c='gray')
    plt.title(f'bg_s_encoder KL coefficient = {kl_coefficient}', fontdict={'fontsize': 'xx-large'})
    plt.show()
    fault_train_set_pre, normal_train_set_pre = cvae.predict([fault_train_set, normal_train_set], batch_size=batch_size)

    fault_train_set_pre = fault_train_set_pre.reshape(fault_train_set_pre.shape[0] * fault_train_set_pre.shape[1])
    normal_train_set_pre = normal_train_set_pre.reshape(normal_train_set_pre.shape[0] * normal_train_set_pre.shape[1])
    fault_train_set = fault_train_set.reshape(fault_train_set.shape[0] * fault_train_set.shape[1])
    normal_train_set = normal_train_set.reshape(normal_train_set.shape[0] * normal_train_set.shape[1])
    plt.figure()
    plt.title(f'fault_train_set KL coefficient ={kl_coefficient}')
    plt.plot(range(577), fault_train_set[0:577].tolist())
    plt.plot(range(577), fault_train_set_pre[0:577].tolist())
    plt.show()
    plt.figure()
    plt.title(f'normal_train_set KL coefficient ={kl_coefficient}')
    plt.plot(range(577), normal_train_set[0:577].tolist())
    plt.plot(range(577), normal_train_set_pre[0:577].tolist())
    plt.show()
def fit_csmote(normal_train_set, fault_train_set,normal_test_set, fault_test_set, latent_dim,batch_size,epochs, optimizer,kl_coefficient,loss_weight):
    csmote, csmote_fg, z_encoder, s_encoder, csmote_decoder, tg_rec_loss,bg_rec_loss,tg_z_kl_loss,tg_s_kl_loss,bg_s_kl_loss = get_csmote(input_dim=fault_train_set.shape[1],
        latent_dim=latent_dim)
    csmote.summary()
    a1,a2,a3,a4,a5=loss_weight
    reconstruction_loss = K.mean((a1*tg_rec_loss+a2*bg_rec_loss))
    kl_loss =K.mean(-0.5 *K.sum(a3*tg_z_kl_loss+a4*tg_s_kl_loss+a5*bg_s_kl_loss, axis=-1))
    csmote_loss = reconstruction_loss +  kl_coefficient *kl_loss
    csmote.add_loss(cvae_loss)
    csmote.compile(optimizer=optimizer)
    target = fault_train_set
    background = normal_train_set
    history= csmote.fit([target, background], epochs=epochs, batch_size=batch_size,validation_data=([fault_test_set, normal_test_set], None),verbose=1)
    min_loss, min_val_loss =  plot_loss(history, kl_coefficient)
    models = csmote, z_encoder, s_encoder, csmote_decoder
    train = fault_train_set, normal_train_set
    plot_csmote_result(models,train,kl_coefficient,batch_size)
    return csmote, csmote_fg, z_encoder, s_encoder, csmote_decoder
def process_csmote():
    train_min = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/train_min.csv'),
        delimiter=",",
        skiprows=0)
    train_maj = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/train_maj.csv'),
        delimiter=",",
        skiprows=0)
    test_min = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/test_min.csv'),
        delimiter=",",
        skiprows=0)
    test_maj = np.loadtxt(
        open('/usr/CSMOTE/Datasets/origin/Car/test_maj.csv'),
        delimiter=",",
        skiprows=0)
    train_min = np.repeat(train_min, (train_maj.shape[0] / train_min.shape[0])+1, axis=0)  # 对现有的故障数据集进行复制扩充，和正常样本同样的数量
    np.random.shuffle(train_min)
    train_min =train_min[0:train_maj.shape[0],:]

    test_min = np.repeat(test_min, (test_maj.shape[0] / test_min.shape[0]) + 1, axis=0) 
                          
    np.random.shuffle(test_min)
    test_min = test_min[0:test_maj.shape[0], :]

    
def generate():
    h = 4
    n = 49
    a = np.hstack((np.random.normal(0, 1, (n, h)), np.random.normal(0, 1, (n, h))))
    decoder = load_model('/usr/CSMOTE/CSMOTE/Car/csmote_decoder_' + str(h)  + '.h5')
    tg = decoder.predict(a)
    np.savetxt('/usr/CSMOTE/Datasets/synthistic/Car/csmote_' + str(h) + '.csv', tg.reshape((n, 577)), delimiter=',')

