from model import VAE

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "")
flags.DEFINE_float("learning_rate", 3e-4, "")
flags.DEFINE_integer("image_size", 28, "")
flags.DEFINE_integer("channels", 1, "")
flags.DEFINE_integer("n_z", 20, "")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "")
flags.DEFINE_string("tensorboard_dir", "tensorboard", "")
FLAGS = flags.FLAGS

def main(_):

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

  mnist = input_data.read_data_sets('MNIST')
  #REPLACE WITH YOUR OWN CLASS. FUNCTION next_batch

  with tf.Session() as sess:
    vae = VAE(sess, mnist, FLAGS)
    vae.train()

if __name__ == '__main__':
  tf.app.run()
