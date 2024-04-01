#!/usr/bin/env python
# coding: utf-8

# **TensorFlow tuto:** https://www.tensorflow.org/tutorials/generative/cyclegan 

###############==================== **Main libraries** ====================###############
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print(tf.__version__)
tf.config.run_functions_eagerly(True)
import subprocess

#allows to import generator and discriminator
install_command = "pip install -q git+https://github.com/tensorflow/examples.git"
# Use subprocess to run the command
subprocess.run(install_command, shell=True)

import gc
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from os import listdir
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import vstack
from numpy import asarray
from numpy import savez_compressed
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import keras
import gc
AUTOTUNE = tf.data.AUTOTUNE


###############============ **Load and preprocess dataset** ====================######
str2idx = {
    'control': 0,
    'patient': 1
}

idx2str = {
    0: 'control',
    1: 'patient'
}

def ohe_class(index):
    ohe_label = np.zeros(2, dtype=int)
    ohe_label[index] = 1
    return ohe_label

def rgb2gray(path, filename, size, pixels):
    img2 = np.zeros((pixels.shape))
    a = load_img(path + filename, target_size=size, color_mode= "grayscale")
    img2[:,:,0] = a
    img2[:,:,1] = a
    img2[:,:,2] = a

    return img2

# load all images in a directory into memory
def load_images(path, size=(256,256), rgb=True):
    data_list = list()
    label_list = list()

    # enumerate filenames in directory, assume all are images
    for filename in tqdm(os.listdir(path)):
        clase = filename.split('_')[0]

        # load and resize the image
        pixels = load_img(path + filename, target_size=size, color_mode= "rgb")
        # convert to numpy array
        pixels = img_to_array(pixels)

        if rgb==False:
            #convert rgb to gray
            pixels = rgb2gray(path, filename, size, pixels)
        else:
            None
        
        # store
        data_list.append(pixels)

        #for labels sub-control032059_output-slice000, sub-patient032078_output-slice000
        clase = filename.split('_')[0]
        if "sub-control" in clase:
            clase = "control"
        else:
            clase = "patient"
        indx = str2idx[clase]
        #get ohe from index
        ohe_label = ohe_class(indx)
        label_list.append(ohe_label)

    return asarray(data_list), label_list

###############============ **Setup** ====================######

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

###############============ **Dataset loading** ====================######

"""Frames loading.
Sets domains: A for control and B for parkinson
rgb parameter sets as True for work whit grayscale images
"""
# dataset path
path = "../data/"
#if we want to load rgb images
rgb = True

# load dataset 
train_control_imgs, train_control_labels = load_images(path + 'train_control/', rgb= rgb)
#test_control_imgs, test_control_labels = load_images(path + 'test_control/', rgb= rgb)

# load dataset B
train_parkinson_imgs, train_parkinson_labels = load_images(path + 'train_parkinson/', rgb= rgb)
#test_parkinson_imgs, test_parkinson_labels = load_images(path + 'test_parkinson/', rgb= rgb)

print("train images control: ", train_control_imgs.shape, " labels: ", len(train_control_labels))
print("train images parkinson: ", train_parkinson_imgs.shape, " labels: ", len(train_parkinson_labels))
# print("test images control: ", test_control_imgs.shape, " labels: ", len(test_control_labels))
# print("test images parkinson: ", test_parkinson_imgs.shape, " labels: ", len(test_parkinson_labels))


#**Data augmentation techniques**

def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

# scaling the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)
    # random mirroring
    image = tf.image.random_flip_left_right(image)
    return image


# Preprocess splits**

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image):
    image = normalize(image)
    return image

###############============ **Tf dataset** ====================######

#conversion de las imageness a array
train_control_array = np.asarray(train_control_imgs)
#test_control_array = np.asarray(test_control_imgs)
train_parkinson_array = np.asarray(train_parkinson_imgs)
#test_parkinson_array = np.asarray(test_parkinson_imgs)

#Crea un dataSet de WL y NBI 
train_control_ds = tf.data.Dataset.from_tensor_slices(train_control_array)
train_control_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_control_labels, tf.int64)).batch(BATCH_SIZE)

train_parkinson_ds = tf.data.Dataset.from_tensor_slices(train_parkinson_array)
train_parkinson_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_parkinson_labels, tf.int64)).batch(BATCH_SIZE)

train_control_ds = train_control_ds.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

train_parkinson_ds = train_parkinson_ds.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

#Since the datasets are in the same order you can just zip them together to get
#a dataset of (image, label) pairs:

train_control_image_label_ds = tf.data.Dataset.zip((train_control_ds, train_control_label_ds))
train_parkinson_image_label_ds = tf.data.Dataset.zip((train_parkinson_ds, train_parkinson_label_ds))

#shuffle zip train data
train_control_image_label_ds = train_control_image_label_ds.shuffle(buffer_size=len(train_control_image_label_ds),
                                                          reshuffle_each_iteration=True)
train_control_image_label_ds = train_control_image_label_ds.prefetch(buffer_size=AUTOTUNE)

train_parkinson_image_label_ds = train_parkinson_image_label_ds.shuffle(buffer_size=len(train_parkinson_image_label_ds),
                                                          reshuffle_each_iteration=True)
train_parkinson_image_label_ds = train_parkinson_image_label_ds.prefetch(buffer_size=AUTOTUNE)

img_sample_control, lab_sample_control  = next(iter(train_control_image_label_ds))
img_sample_parkinson, lab_sample_parkinson = next(iter(train_parkinson_image_label_ds))

print("control sample info:")
print("shape: {}, label: {} ".format(img_sample_control.shape, lab_sample_control))
print("parkinson sample info:")
print("shape: {}, label: {} ".format(img_sample_parkinson.shape, lab_sample_parkinson))

###############============ **Here I go** ====================######
b = train_parkinson_array[0]
plt.hist(b.ravel())
plt.title("Before scaling")
plt.show()
print("min: {}, max:{}".format(b.min(), b.max()))


# In[ ]:


a = np.array(img_sample_parkinson[0])
plt.hist(a.ravel())
plt.title("After scaling")
plt.show()
print("min: {}, max:{}".format(a.min(), a.max()))


# In[ ]:


plt.subplot(121)
plt.title('control')
#plt.imshow(sample_WL[0] * 0.5 + 0.5)
print(img_sample_control[0].shape)
plt.imshow(np.squeeze(img_sample_control[0]) * 0.5 + 0.5, cmap='gray')
idx = lab_sample_control.numpy().argmax()
plt.xlabel(idx2str[idx])

plt.subplot(122)
plt.title('control with random jitter')
#plt.imshow(random_jitter(sample_WL[0]) * 0.5 + 0.5)
plt.imshow(np.squeeze(random_jitter(img_sample_control[0])) * 0.5 + 0.5, cmap='gray')
idx = lab_sample_control.numpy().argmax()
plt.xlabel(idx2str[idx])


# In[ ]:


plt.subplot(121)
plt.title('parkinson')
#plt.imshow(sample_NBI[0] * 0.5 + 0.5)
plt.imshow(np.squeeze(img_sample_parkinson[0]) * 0.5 + 0.5, cmap='gray')
idx = lab_sample_parkinson.numpy().argmax()
plt.xlabel(idx2str[idx])

plt.subplot(122)
plt.title('parkinson with random jitter')
#plt.imshow(random_jitter(sample_NBI[0]) * 0.5 + 0.5)
plt.imshow(np.squeeze(random_jitter(img_sample_parkinson[0])) * 0.5 + 0.5, cmap='gray')
idx = lab_sample_parkinson.numpy().argmax()
plt.xlabel(idx2str[idx])


# **Loading many parkinson samples**

# In[ ]:


images, labels = [], []
for i in range(25):
    imgs_samples, labels_samples = next(iter(train_parkinson_image_label_ds.shuffle(buffer_size=len(train_parkinson_imgs))))
    images.append(imgs_samples)
    labels.append(labels_samples)

images = np.asarray(images)


# In[ ]:


plt.figure(figsize=(12,12))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(np.squeeze(images[i])* 0.5 + 0.5)#convert (batch, high, width, #channels) into (high, width, #channels) 
    idx = labels[i].numpy().argmax()
    plt.xlabel("label: {}".format(idx2str[idx]))
plt.show()


# # Import and reuse the Pix2Pix models

# In[ ]:


# OUTPUT_CHANNELS = 3

# generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
# generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# ## Improving the Generator model
# ### Basic utils

# In[ ]:


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


# In[ ]:


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


# In[ ]:


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


# ### Generators

# In[ ]:


def unet_generator(img_width, img_high, output_channels, norm_type='instancenorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
  Returns:
    Generator model
  """

  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
      downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
      downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
      downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2,
                                         padding='same', kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)

  img_input = tf.keras.layers.Input(shape=[img_width, img_high, 3])

  x = img_input

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    #x = concat([x, skip])
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=img_input, outputs=x) #tf.keras.Model(inputs=[img_input, input_label], outputs=x)


# In[ ]:


OUTPUT_CHANNELS = 3
generator_g = unet_generator(img_width=IMG_WIDTH, img_high=IMG_HEIGHT, output_channels=OUTPUT_CHANNELS)
generator_f = unet_generator(img_width=IMG_WIDTH, img_high=IMG_HEIGHT, output_channels=OUTPUT_CHANNELS)


# ## Until here

# In[ ]:


img_sample_control, label_sample_control = next(iter((train_control_image_label_ds)))
img_sample_parkinson, label_sample_parkinson = next(iter((train_parkinson_image_label_ds)))

print("info de real data")
print("img shape: {}, label: {}".format(img_sample_control.shape, lab_sample_control))
print("min: {}, max: {}".format(tf.reduce_min(img_sample_control).numpy(), tf.reduce_max(img_sample_control).numpy()))
print("min: {}, max: {}".format(tf.reduce_min(img_sample_parkinson).numpy(), tf.reduce_max(img_sample_parkinson).numpy()))

to_parkinson = generator_g([img_sample_control])
to_control = generator_f([img_sample_parkinson])

print("info de fake data")
print("min: {}, max: {}".format(tf.reduce_min(to_parkinson).numpy(), tf.reduce_max(to_parkinson).numpy()))
print("min: {}, max: {}".format(tf.reduce_min(to_control).numpy(), tf.reduce_max(to_control).numpy()))

plt.figure(figsize=(8, 8))
contrast = 8

imgs = [img_sample_control, to_parkinson, img_sample_parkinson, to_control]
title = ['control', 'To parkinson', 'parkinson', 'To control']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()


# In[ ]:


print(img_sample_parkinson.shape)
print(label_sample_parkinson.shape)


# In[ ]:


plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real parkinson?')
plt.imshow(discriminator_y([img_sample_parkinson])[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real control?')
plt.imshow(discriminator_x([img_sample_control])[0, ..., -1], cmap='RdBu_r')

plt.show()


# ## **Loss functions**

# In[ ]:


LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
class_loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# In[ ]:


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


# ## **Initializing optimizers, generator and discriminators**

# In[ ]:


lr = 2e-4
generator_g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)


# ## **Check points**

# In[ ]:


checkpoint_path = "../../models/translation/unet_fixed/"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

ckpt.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


# # **Training**

# In[ ]:


EPOCHS = 50


# In[ ]:


def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


# In[ ]:


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    real_x_img = real_x[0]
    real_x_label = real_x[1]
    real_y_img = real_y[0]
    real_y_label = real_y[1]
    
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y------> WL -> NBI
        # Generator F translates Y -> X.-----> NBI -> WL

        fake_y = generator_g(real_x_img, training=True)
        cycled_x = generator_f(fake_y, training=True)
        #same for revert domain traslation
        fake_x = generator_f(real_y_img, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x_img, training=True)
        same_y = generator_g(real_y_img, training=True)

        disc_real_x = discriminator_x(real_x_img, training=True)
        disc_real_y = discriminator_y(real_y_img, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss (generator)
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)
            
        total_cycle_loss = calc_cycle_loss(real_x_img, cycled_x) + calc_cycle_loss(real_y_img, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        #what happened if the identity loss is not taken?
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y_img, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x_img, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
        

    ### Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)
        

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


# In[ ]:


def train_and_checkpoint(ckpt_manager=None):
    
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in range(EPOCHS):
        start = time.time()
        n = 0    

        for image_x, image_y in tf.data.Dataset.zip((train_control_image_label_ds, train_parkinson_image_label_ds)):
            train_step(image_x, image_y)
            if n % 10 == 0:
                print ('.', end='')
            n += 1

        clear_output(wait=True)
        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        generate_images(generator_g, img_sample_control)

        if (epoch + 1) % 5 == 0:
                       
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            #print('Saving generator g h5 models...')
            #generator_g_name = save_model_h5_path + '/gen_g.h5'
            #
            #generator_g.save(generator_g_name)
            #print('Saving classification net')
            #cls_model_name = save_model_h5_path + '/nbi_clss.h5'
            #nbi_cls_model.save(cls_model_name)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


# In[ ]:


train_and_checkpoint(ckpt_manager)


# # Testing over single video

# In[ ]:


def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()# **Generate using test dataset**


# In[ ]:


# frames from video path
path =  "../../../../../data/polyp_original/WL/adenoma_WL/video_1/"
# load dataset white light
# here A: white light, B: nbi light
adenoma_WL = load_images(path, rgb=True)
print("Adenoma WL video_1: ", adenoma_WL.shape)


# In[ ]:


adenoma_WL_array = np.asarray(adenoma_WL)
adenoma_WL_ds = tf.data.Dataset.from_tensor_slices(adenoma_WL_array)
adenoma_WL_ds = adenoma_WL_ds.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
                BUFFER_SIZE).batch(BATCH_SIZE)


# In[ ]:


for inp in adenoma_WL_ds.take(adenoma_WL.shape[0]):
    generate_images(generator_g, inp)


# In[ ]:




