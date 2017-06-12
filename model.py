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
    self.numLatent = flags.numLatent
    self.tensorboard_dir = flags.tensorboard_dir
    self.checkpoint_dir = flags.checkpoint_dir
    self.learning_rate = flags.learning_rate
    self.saveModel = flags.saveModel
    self.loadModel = flags.loadModel
    self.saveData = flags.saveData
    self.saveRate = flags.saveRate
    self.model()
    
  def model(self):

    #RESHAPING
    self.images = tf.placeholder(tf.float32, [None, None, None, self.channels])
    reshapedImages = self.images
    if(reshapedImages.shape[1] > reshapedImages.shape[2]):
      reshapedImages = tf.image.crop_to_bounding_box(reshapedImages, (reshapedImages.shape[1]-reshapedImages.shape[2])/2, 0, reshapedImages.shape[2], reshapedImages.shape[2])
    if(reshapedImages.shape[2] > reshapedImages.shape[1]):
      reshapedImages = tf.image.crop_to_bounding_box(reshapedImages, 0, (reshapedImages.shape[2]-reshapedImages.shape[1])/2, reshapedImages.shape[1], reshapedImages.shape[1])
    image_matrix = tf.image.resize_images(reshapedImages, [self.image_size, self.image_size])
    images_flat = tf.reshape(image_matrix, [self.batch_size, self.image_size*self.image_size*self.channels])

    #RECOGNITION, LATENT, GENERATION
    z_mean, z_stddev = self.recognition(image_matrix)
    samples = tf.random_normal([self.batch_size,self.numLatent], 0, 1, dtype=tf.float32)
    guessed_z = z_mean + (z_stddev * samples)
    generated_images = self.generation(guessed_z)
    generated_flat = tf.reshape(generated_images, [self.batch_size, self.image_size*self.image_size*self.channels])

    #LOSS AND OPTIMIZER
    generation_loss = -tf.reduce_sum(images_flat * tf.log(1e-8 + generated_flat) + (1-images_flat) * tf.log(1e-8 + 1 - generated_flat), 1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
    cost = tf.reduce_mean(generation_loss + latent_loss)
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

    #SAVER AND TENSORBOARD DATA
    BCESum = tf.summary.scalar('generation loss', tf.reduce_mean(generation_loss))
    KLDSum = tf.summary.scalar('latent loss', tf.reduce_mean(latent_loss))
    LossSum = tf.summary.scalar('cost', cost)
    inImg = tf.summary.image("original", image_matrix)
    outImg = tf.summary.image("reconstructed", generated_images)
    self.sum = tf.summary.merge_all()
    self.saver = tf.train.Saver()

  #Take care to ensure dimensions sizes are set correctly when altering the model

  #ENCODER
  def recognition(self, input_images):
      with tf.variable_scope("recognition"):
          h1 = tf.nn.relu(conv2d(input_images, self.channels, 16, 5, 2, "d_h1"))
          h2 = tf.nn.relu(conv2d(h1, 16, 32, 5, 2, "d_h2"))
          h2_flat = tf.reshape(h2,[self.batch_size, self.image_size//4*self.image_size//4*32])

          w_mean = dense(h2_flat, self.image_size//4*self.image_size//4*32, self.numLatent, "w_mean")
          w_stddev = dense(h2_flat, self.image_size//4*self.image_size//4*32, self.numLatent, "w_stddev")

      return w_mean, w_stddev

  #DECODER
  def generation(self, z):
      with tf.variable_scope("generation"):
          z_develop = dense(z, self.numLatent, self.image_size//4*self.image_size//4*32, scope='z_matrix')
          z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batch_size, self.image_size//4, self.image_size//4, 32]))
          h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batch_size, self.image_size//2, self.image_size//2, 16], 5, 2, "g_h1"))
          h2 = conv_transpose(h1, [self.batch_size, self.image_size, self.image_size, self.channels], 5, 2, "g_h2")
          h2 = tf.nn.sigmoid(h2)

      return h2


  def train(self):
    board_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)
    tf.global_variables_initializer().run()

    if self.loadModel:
      if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")

    print("Start training...")
    
    counter = 0
    while True:
      batch = self.input_data.next_batch(self.batch_size)
      self.sess.run(self.optimizer, feed_dict={self.images: batch})
      if counter % self.saveRate == 0:
        print('batch {}'.format(counter))
        if self.saveData:
          summary = self.sess.run(self.sum, feed_dict={self.images: batch})
          board_writer.add_summary(summary, counter)
        if self.saveModel:
          self.save(self.checkpoint_dir, counter)
      counter += 1

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
