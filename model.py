import scipy.misc
import time
import os

import numpy as np
import tensorflow as tf
from ops import *

class VAE(object):

  def __init__(self, sess, input_data, flags):

    self.sess = sess
    self.input_data = input_data
    self.batch_size = flags.batch_size
    self.image_size = flags.image_size
    self.channels = flags.channels
    self.n_z = flags.n_z
    self.tensorboard_dir = flags.tensorboard_dir
    self.checkpoint_dir = flags.checkpoint_dir
    self.model()
    
  def model(self):

    #self.n_hidden = 500
    #self.n_z = 20
    #self.batchsize = 1

    self.images = tf.placeholder(tf.float32, [None, 784])
    #self.images = tf.placeholder(tf.float32, [None, None, None, self.channels])
    #reshapedImages = self.images
    #if(reshapedImages.shape[1] > reshapedImages.shape[2]):
    #  reshapedImages = tf.image.crop_to_bounding_box(reshapedImages, (reshapedImages.shape[1]-reshapedImages.shape[2])/2, 0, reshapedImages.shape[2], reshapedImages.shape[2])
    #if(reshapedImages.shape[2] > reshapedImages.shape[1]):
    #  reshapedImages = tf.image.crop_to_bounding_box(reshapedImages, 0, (reshapedImages.shape[2]-reshapedImages.shape[1])/2, reshapedImages.shape[1], reshapedImages.shape[1])
    #scaledImages = tf.image.resize_images(reshapedImages, [self.image_size, self.image_size])
    image_matrix = tf.reshape(self.images,[-1, self.image_size, self.image_size, self.channels])

    z_mean, z_stddev = self.recognition(image_matrix)
    samples = tf.random_normal([self.batch_size,self.n_z],0,1,dtype=tf.float32)
    guessed_z = z_mean + (z_stddev * samples)
    generated_images = self.generation(guessed_z)
    generated_flat = tf.reshape(generated_images, [self.batch_size, 28*28])

    generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
    cost = tf.reduce_mean(generation_loss + latent_loss)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    BCESum = tf.summary.scalar('generation loss', tf.reduce_mean(generation_loss))
    KLDSum = tf.summary.scalar('latent loss', tf.reduce_mean(latent_loss))
    LossSum = tf.summary.scalar('cost', cost)

    inImg = tf.summary.image("original", image_matrix)
    outImg = tf.summary.image("reconstructed", generated_images)

    self.sum = tf.summary.merge_all()
    self.saver = tf.train.Saver()

  # encoder
  def recognition(self, input_images):
      with tf.variable_scope("recognition"):
          h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
          h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
          h2_flat = tf.reshape(h2,[self.batch_size, 7*7*32])

          w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
          w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

      return w_mean, w_stddev

  # decoder
  def generation(self, z):
      with tf.variable_scope("generation"):
          z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
          z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, 7, 7, 32]))
          h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, 14, 14, 16], "g_h1"))
          h2 = conv_transpose(h1, [self.batch_size, 28, 28, 1], "g_h2")
          h2 = tf.nn.sigmoid(h2)

      return h2


  def train(self):

    board_writer = tf.summary.FileWriter(self.tensorboard_dir)
    tf.global_variables_initializer().run()
    counter = 0

    #if self.load(self.checkpoint_dir):
    #  print(" [*] Load SUCCESS")
    #else:
    #  print(" [!] Load failed...")

    print("Start training...")

    while True:
      batch = self.input_data.train.next_batch(self.batch_size)
      counter += 1
      _, summary = self.sess.run([self.optimizer, self.sum], feed_dict={self.images: batch[0]})
      board_writer.add_summary(summary, counter)

      if counter % 5000 == 0:
        self.save(self.checkpoint_dir, counter)

  def save(self, checkpoint_dir, step):
    model_name = "vae.model"
    model_dir = "{}".format("vae")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, 
                    os.path.join(checkpoint_dir, model_name), 
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "{}".format("vae")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False
