from LoadData import *
from FastSCNN import FSCNN
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from LabelImages import *


data = TapeRoad()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
val_x, val_y = data.get_val_data(batch_size=1)
input_data = val_x.astype('float32')
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
display_image(val_x[0])
output_data = interpreter.get_tensor(output_details[0]['index'])
cat_image = np.argmax(output_data[0], axis=-1)
display_image(cat_to_im(cat_image))
