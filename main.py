from model import VAE
from feeder import webcamFeeder, videoFeeder
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

  #feed = input_data.read_data_sets('MNIST')
  feed = webcamFeeder()
  #feed = videoFeeder("Disco_Hot_Dog_Pyramid.mp4")

  with tf.Session() as sess:
    vae = VAE(sess, feed, FLAGS)
    vae.train()

if __name__ == '__main__':
  tf.app.run()
