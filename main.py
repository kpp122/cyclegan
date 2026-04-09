from os import listdir
from matplotlib import pyplot
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy.random import randint, random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from keras.models import Model, load_model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from numpy import savez_compressed, expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import vstack
from numpy import load


# load all data, size used is 256, 256
def load_images(path, size=(256, 256)):
    data_list = list()
    for filename in listdir(path):
        # load & resize
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)

    return asarray(data_list)


def compress_images(path):
    # load dataset A
    dataA1 = load_images(path + 'trainA/')
    dataAB = load_images(path + 'testA/')
    dataA = vstack((dataA1, dataAB))
    print('Loaded dataA: ', dataA.shape)
    # load dataset B
    dataB1 = load_images(path + 'trainB/')
    dataB2 = load_images(path + 'testB/')
    dataB = vstack((dataB1, dataB2))
    print('Loaded dataB: ', dataB.shape)
    # save as compressed numpy array
    filename = 'photo2vangogh_256.npz'
    savez_compressed(filename, dataA, dataB)
    print('Saved dataset: ', filename)


def randomtest():
    data = load('photo2vangogh_256.npz')
    dataA, dataB = data['arr_0'], data['arr_1']
    print('Loaded: ', dataA.shape, dataB.shape)
    # plot source images
    n_samples = 3
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(dataA[i].astype('uint8'))
    # plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(dataB[i].astype('uint8'))
    pyplot.show()


## MODELS
def discriminator(img_shape):
    # weights
    init = RandomNormal(stddev=0.02)
    # source img
    in_image = Input(shape=img_shape)
    # Layer 64
    discriminator = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    # Layer 128
    discriminator = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(discriminator)
    discriminator = InstanceNormalization(axis=-1)(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    # Layer 256
    discriminator = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(discriminator)
    discriminator = InstanceNormalization(axis=-1)(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    # Layer 512
    discriminator = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(discriminator)
    discriminator = InstanceNormalization(axis=-1)(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    # Layer 512
    discriminator = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(discriminator)
    discriminator = InstanceNormalization(axis=-1)(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    # Output layer
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(discriminator)

    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def residual_network_block(filters_amount, input_layer):
    # Weights
    init = RandomNormal(stddev=0.02)
    # First CN layer
    block = Conv2D(filters_amount, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    block = InstanceNormalization(axis=-1)(block)
    block = Activation('relu')(block)
    # Second CN layer
    block = Conv2D(filters_amount, (3, 3), padding='same', kernel_initializer=init)(block)
    block = InstanceNormalization(axis=-1)(block)

    block = Concatenate()([block, input_layer])
    return block


def generator(img_shape, resnet_amount=9):
    # Weights
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=img_shape)

    generator = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)

    generator = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)

    generator = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)

    # How many ResNet blocks?
    for _ in range(resnet_amount):
        generator = residual_network_block(256, generator)

    generator = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)

    generator = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)

    generator = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    out_image = Activation('tanh')(generator)

    model = Model(in_image, out_image)
    return model


# Model for updating generators by adversarial and cycle loss
def define_composite_model(gener_model_1, disc_model, gener_model_2, image_shape):
    gener_model_1.trainable = True
    disc_model.trainable = False
    gener_model_2.trainable = False

    # Dics_model element
    input_gen = Input(shape=image_shape)
    gen1_out = gener_model_1(input_gen)
    output_d = disc_model(gen1_out)

    input_id = Input(shape=image_shape)
    output_id = gener_model_1(input_id)
    # cycle forward
    output_f = gener_model_2(gen1_out)
    # cycle backwards
    gen2_out = gener_model_2(input_id)
    output_b = gener_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    # Optimization with adam (test different lr)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model


## DATA PREPARATION

# load samples from npz
def load_samples(file):
    data = load(file)
    # unpack photos (X1) and targets (paintings, X2)
    X1, X2 = data['arr_0'], data['arr_1']
    # [0,255] -> [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# real images for discriminator and composite models input
def generate_samples(dataset, samples_amount, patch_shape):
    # select instances by random and label them as real images
    ix = randint(0, dataset.shape[0], samples_amount)
    X = dataset[ix]
    y = ones((samples_amount, patch_shape, patch_shape, 1))
    return X, y


# generated FAKE samples to update discriminators
def generate_fake_samples(gene_model, dataset, patch_shape):
    # use generative model to predict, so its fake
    X = gene_model.predict(dataset)
    # label them as fakes
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# save generator models, for testing purposes
def save_models(step, gene_model_AtoB, gene_model_BtoA):
    filename1 = 'g_model_AtoB_%06d.h5' % (step + 1)
    gene_model_AtoB.save(filename1)

    filename2 = 'g_model_BtoA_%06d.h5' % (step + 1)
    gene_model_BtoA.save(filename2)

    print('>Saved: %s and %s' % (filename1, filename2))


def summarize_performance(step, g_model, trainX, name, samples_amount=5):
    # select a sample of input images
    X_in, _ = generate_samples(trainX, samples_amount, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(samples_amount):
        pyplot.subplot(2, samples_amount, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])
    # plot translated image
    for i in range(samples_amount):
        pyplot.subplot(2, samples_amount, 1 + samples_amount + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])
    # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step + 1))
    pyplot.savefig(filename1)
    pyplot.close()


# update fake image pool for discriminator
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)


# CycleGAN training
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    epochs, batch, = 100, 1
    patch = d_model_A.output_shape[1]
    trainA, trainB = dataset
    poolA, poolB = list(), list()
    batches_per_epoch = int(len(trainA) / batch)
    print(len(trainA))
    steps = batches_per_epoch * epochs

    for i in range(steps):
        # generated batches of REAL samples per each step
        X_realA, y_realA = generate_samples(trainA, batch, patch)
        X_realB, y_realB = generate_samples(trainB, batch, patch)
        # generated batches of FAKE samples per each step
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
        i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        # evaluate the model performance every so often
        if (i + 1) % (102 * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i + 1) % (102 * 1) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)


# FOR PREDICTING IMAGES
def load_predict_images(path, size=(256, 256)):
    data = list()
    for filename in listdir(path):
        # load & resize
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        pixels = expand_dims(pixels, 0)
        pixels = (pixels - 127.5) / 127.5

        data.append(pixels)
    return asarray(data)


def predict_kuopio(path, g_model_AtoB, g_model_BtoA):
    data = load_predict_images(path)
    step = 0

    for img in data:
        real = img
        generated = g_model_AtoB.predict(real)
        reconstructed = g_model_BtoA(generated)

        images = vstack((real, generated, reconstructed))
        titles = ['Real', 'Generated', 'Reconstructed']
        # scale from [-1,1] to [0,1]
        images = (images + 1) / 2.0

        for i in range(len(images)):

            pyplot.subplot(1, len(images), 1 + i)
            pyplot.axis('off')
            pyplot.imshow(images[i])
            pyplot.title(titles[i])

        filename = '%s_generated_plot_2_%06d.png' % ('kuopio', (step + 10))
        pyplot.savefig(filename)
        pyplot.close()
        step += 1
# /FOR PREDICTING IMAGES


if __name__ == '__main__':
    # Preparing images
    # compress_images('photo2vangogh/')

    # Images for training
    dataset = load_samples('photo2vangogh_256_2.npz')
    image_shape = dataset[0].shape[1:]

    # generators for source to target & target to source
    '''g_model_AtoB = generator(image_shape)
    g_model_BtoA = generator(image_shape)'''
    # OR if continuing training from old models
    cust = {'InstanceNormalization': InstanceNormalization}
    g_model_AtoB = load_model('g_model_AtoB_1_000800.h5', cust)
    g_model_BtoA = load_model('g_model_BtoA_1_000800.h5', cust)
    g_model_AtoB._name = 'model_1'
    g_model_BtoA._name = 'model_2'

    # FOR IMAGE PREDICTION
    #predict_kuopio('kuopio/', g_model_AtoB, g_model_BtoA)

    # discriminators for A and B to determine if fake or not
    d_model_A = discriminator(image_shape)
    d_model_B = discriminator(image_shape)
    # composite generators & discriminators
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    # train models
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)

