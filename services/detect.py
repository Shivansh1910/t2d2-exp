"""A module to run object detection with a TensorFlow Lite model."""

import platform
from typing import List, NamedTuple
import zipfile

import numpy as np


try:
  # Import TFLite interpreter from tflite_runtime package
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  
  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate

num_threads = 4
enable_edgetpu = False 

model = 'models/model.tflite'

