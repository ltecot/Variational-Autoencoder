from model import VAE
from feeder import webcamFeeder, videoFeeder
import numpy as np
import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "Number of images taken in per step.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer coefficient")
flags.DEFINE_integer("image_size", 64, "Length of image size. Input with be squared and shrunk to image_size*image_size")
flags.DEFINE_integer("channels", 3, "Number of color channels in the images. Change to 3 if you use color in your feeder, 1 otherwise.")
flags.DEFINE_integer("numLatent", 20, "Size of the latent, stddev and mean vectors")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory where models are saved")
flags.DEFINE_string("tensorboard_dir", "tensorboard", "Directory where tensorboard data is saved")
flags.DEFINE_string("saveModel", True, "Whether to save model progress or not")
flags.DEFINE_string("loadModel", True, "Whether to load the previous model or not")
flags.DEFINE_string("saveData", True, "Whether to save model data or not")
flags.DEFINE_string("saveRate", 2500, "Step interval at which to save the model")
FLAGS = flags.FLAGS

def main(_):

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.tensorboard_dir):
    os.makedirs(FLAGS.tensorboard_dir)

  feed = webcamFeeder(color=True)
  #feed = videoFeeder("Disco_Hot_Dog_Pyramid.mp4", color=True)

  with tf.Session() as sess:
    vae = VAE(sess, feed, FLAGS)
    vae.train()

if __name__ == '__main__':
  tf.app.run()
