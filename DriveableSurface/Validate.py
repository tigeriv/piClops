from LoadData import *
from FastSCNN import FSCNN
from tensorflow.python.ops import variables
from tensorflow.python.framework import ops
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from LabelImages import *

import matplotlib.pyplot as plt


load_path = "./tmp/model600.ckpt"
save = True
load = True
lite = True


def load_sess(path, model):
    sess = tf.Session(graph=model.graph)
    model.saver.restore(sess, path)
    return sess


def predict(sess, model, images):
    feed_dict = {model.X: images}
    outs = sess.run([model.predictions], feed_dict=feed_dict)[0]
    sess_temp = tf.Session()
    with sess_temp.as_default():
        labels = tf.nn.softmax(outs).eval(session=sess_temp)
    return labels


def save_graph(model):
    with tf.Session(graph=model.graph) as sess:
        model.saver.restore(sess, load_path)
        saver = tf.train.Saver()
        saver.save(sess, './frozen/tensorflowModel.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), './frozen', 'tensorflowModel.pbtxt', as_text=True)


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def to_lite():
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "./frozen/frozen_model.pb", ['Placeholder'], ['Softmax'], input_shapes={'Placeholder': [1, 240, 240, 3]})
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


def printTensors(pb_file):

    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)


if __name__ == "__main__":
    model = FSCNN(width=240, height=240, training=False)
    data = TapeRoad()

    printTensors("frozen/frozen_model.pb")

    if save:
        save_graph(model)
        freeze_graph("frozen", "Softmax")

    if load:
        graph = load_graph("frozen/frozen_model.pb")
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            print(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions

        # We access the input and output nodes
        x = graph.get_tensor_by_name('prefix/Placeholder:0')
        y = graph.get_tensor_by_name('prefix/Softmax:0')

        # We launch a Session
        with tf.Session(graph=graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            while True:
                val_x, val_y = data.get_val_data(batch_size=1)
                y_out = sess.run(y, feed_dict={
                    x: val_x  # < 45
                })
                cat_image = np.argmax(y_out[0], axis=-1)
                display_image(val_x[0])
                display_image(cat_to_im(cat_image))

    if lite:
        to_lite()

    sess = load_sess(load_path, model)
    while True:
        val_x, val_y = data.get_val_data(batch_size=1)
        val_labels = predict(sess, model, val_x)
        for index in range(len(val_labels)):
            display_image(val_x[index])
            cat_image = np.argmax(val_labels[index], axis=-1)
            display_image(cat_to_im(cat_image))