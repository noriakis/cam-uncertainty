import tensorflow as tf
import cv2
import keras
import numpy as np
from PIL import Image

def export_dropout_vgg16(dropout):
    raw = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')
    x = raw.layers[0].output
    for l in raw.layers[1:]:
        if "block5_conv2" in l.name or "fc2" in l.name:
            x = raw.get_layer(l.name)(x)
            x  = tf.keras.layers.Dropout(dropout)(x, training=True)
        else:
            x = raw.get_layer(l.name)(x)
    new_model = tf.keras.Model(inputs= raw.input, outputs=x)
    new_model.save("dropout_model.h5")
    return 0

def GradCam(input_model, image, category_index, layer_name, raw_array, dimension):
    gradModel = tf.keras.Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        (convOuts, preds) = gradModel(inputs)
        loss = preds[:, category_index]
    grads = tape.gradient(loss, convOuts)

    convOuts = convOuts[0]
    grads = grads[0]
    norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)))
    weights = tf.reduce_mean(norm_grads, axis=(0, 1))

    cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
    cam = cam.numpy()
    cam = cv2.resize(cam, (dimension, dimension))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    heatmap = cam.copy()

    image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    cam = cv2.applyColorMap(np.uint8(255-255*cam), cv2.COLORMAP_JET)
    cam = cv2.addWeighted(cam, 0.5, image, 0.5, 0)

    return np.uint8(cam), heatmap


def GradCam_Dropout(input_model, image, category_index, layer_name, raw_array, dimension, sample):
    gradModel = tf.keras.Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(layer_name).output, input_model.output]
        )
    cams = []
    for i in range(0, sample):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) 
            loss = preds[:, category_index]
        grads = tape.gradient(loss, convOuts)
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)))
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cams.append(cam)

    cams = np.asarray(cams)
    m = np.nanmean(cams, axis=0)
    std = np.nanstd(cams, axis=0)
    cov = np.nan_to_num(std / m)

    w_cam = m * (1-cov)
    w_cam = cv2.resize(w_cam, (dimension, dimension),cv2.INTER_LINEAR)
    w_cam = np.maximum(w_cam, 0)
    w_cam = w_cam / np.max(w_cam)

    heatmap = w_cam.copy()

    image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    w_cam = cv2.applyColorMap(np.uint8(255-255*w_cam), cv2.COLORMAP_JET)
    w_cam = cv2.addWeighted(w_cam, 0.5, image, 0.5, 0)

    return np.uint8(w_cam), heatmap
