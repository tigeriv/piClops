from LoadData import *
from FastSCNN import FSCNN
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from LabelImages import *
import imageio


data = TapeRoad()


def load_lite():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def get_mask(input_images, interpreter, input_details, output_details):
    input_data = val_x.astype('float32')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    cat_image = np.argmax(output_data[0], axis=-1)
    return cat_image


def load_img(file_name):
    img = imageio.imread(file_name)
    return img


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
if __name__ == "__main__":
    val_x = np.asarray([load_img("OG.jpg")])
    interpreter, input_details, output_details = load_lite()
    input_shape = input_details[0]['shape']
    cat_image = get_mask(val_x, interpreter, input_details, output_details)
    display_image(val_x[0])
    display_image(cat_to_im(cat_image))
