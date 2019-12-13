"""
modified from https://github.com/nyoki-mtl/keras-facenet/
"""
import os
import re
from typing import NoReturn
import sys
import numpy as np
import tensorflow
if tensorflow.__version__.startswith("1."):
    del tensorflow
    import tensorflow as tf
else:
    del tensorflow
    import tensorflow.compat.v1 as tf 
    tf.disable_v2_behavior()

from src.models.keras_inception_resnet_v1 import InceptionResNetV1


def tf_to_keras(tf_dir:str, tf_model_name:str, ckpt_name:str, embedding_dim:int, keras_dir:str) -> NoReturn:
    """
    """
    tf_model_dir = os.path.join(tf_dir, tf_model_name)
    npy_weights_dir = os.path.join(keras_dir, 'tmp-{}/npy_weights/'.format(tf_model_name))
    weights_dir = os.path.join(keras_dir, 'tmp-{}/weights/'.format(tf_model_name))
    model_dir = os.path.join(keras_dir, 'tmp-{}/model/'.format(tf_model_name))

    weights_filename = 'facenet_keras_weights-{}.h5'.format(tf_model_name)
    model_filename = 'facenet_keras-{}.h5'.format(tf_model_name)

    os.makedirs(npy_weights_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    _extract_tensors_from_checkpoint_file(os.path.join(tf_model_dir, ckpt_name), npy_weights_dir)

    model = InceptionResNetV1(classes=embedding_dim)

    print('Loading numpy weights from', npy_weights_dir)
    for layer in model.layers:
        if layer.weights:
            weights = []
            for w in layer.weights:
                weight_name = os.path.basename(w.name).replace(':0', '')
                weight_file = layer.name + '_' + weight_name + '.npy'
                weight_arr = np.load(os.path.join(npy_weights_dir, weight_file))
                weights.append(weight_arr)
            layer.set_weights(weights)

    print('Saving weights...')
    model.save_weights(os.path.join(weights_dir, weights_filename))
    print('Saving model...')
    model.save(os.path.join(model_dir, model_filename))


def _get_filename(key):
    """
    """
    re_repeat = re.compile(r'Repeat_[0-9_]*b')
    re_block8 = re.compile(r'Block8_[A-Za-z]')

    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('InceptionResnetV1_', '')

    # remove "Repeat" scope from filename
    filename = re_repeat.sub('B', filename)

    if re_block8.match(filename):
        # the last block8 has different name with the previous 5 occurrences
        filename = filename.replace('Block8', 'Block8_6')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def _extract_tensors_from_checkpoint_file(filename, output_folder):
    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # not saving the following tensors
        if key == 'global_step':
            continue
        if 'AuxLogit' in key:
            continue

        # convert tensor name into the corresponding Keras layer weight name and save
        path = os.path.join(output_folder, _get_filename(key))
        arr = reader.get_tensor(key)
        np.save(path, arr)
