import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_overlap(annot, cam):
    inside_annot = (annot.astype("float32") * cam.astype("float32")).sum()
    cam_sum = cam.sum()
    ovl = 0 if cam_sum==0 else inside_annot / cam_sum
    return ovl

def resize_normalize_cam(cam, dimension):    
    cam = cv2.resize(cam, (dimension, dimension))
    cam = np.maximum(cam, 0)
    cam = cam if cam.sum==0 else cam / np.max(cam)
    return cam

def load_bayesian_CNN_1d():
    dropout=0.1
    input_layer = tf.keras.layers.Input((1000,12))
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding="causal", activation="relu")(input_layer)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    x = tf.keras.layers.Conv1D(filters=20, kernel_size=5, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    x = tf.keras.layers.Conv1D(filters=20, kernel_size=5, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    x = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    x = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding="causal", activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    gap = tf.keras.layers.GlobalAveragePooling1D()(x)
    output_layer = tf.keras.layers.Dense(4, activation="softmax")(gap)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.load_weights("weights/best_weights_ECG.h5")
    return model

def load_chexnet_with_dropout():
    img_input = tf.keras.Input(shape=(224,224,3))
    dropout=0.5
    base_model = tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        input_tensor=img_input,
        input_shape=(224,224,3),
        weights="imagenet",
        pooling="avg"
        )
    lastconv = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer("conv5_block16_1_relu").output)
    dout1  = tf.keras.layers.Dropout(dropout)(lastconv.output, training=True)
    x = base_model.get_layer("conv5_block16_2_conv")(dout1)
    x = base_model.get_layer("conv5_block16_concat")([base_model.get_layer("conv5_block15_concat").output, x])
    x = base_model.get_layer("bn")(x)
    x = base_model.get_layer("relu")(x)
    x = base_model.get_layer("avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout)(x, training=True)
    predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = tf.keras.Model(inputs=img_input, outputs=predictions)
    return model

def export_dropout_effnet():
    b0 = tf.keras.applications.EfficientNetB0(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax')

    afterdropout=False
    x = b0.layers[0].output
    for e, l in enumerate(b0.layers[1:]):
        print("current layer:", e+1, l.name)
        if type(l.input)==list and afterdropout:
            x = b0.get_layer(l.name)([x, l.input[1]])
            afterdropout=False
        elif type(l.input)==list and not afterdropout:
            x = b0.get_layer(l.name)([l.input[0], l.input[1]])
        elif "drop" in l.name:
            print("Found dropout, replacing ...", l.rate, l.name)
            x = tf.keras.layers.Dropout(l.rate)(x, training=True)
            afterdropout=True
        else:
            x = b0.get_layer(l.name)(x)
    newmodel = tf.keras.Model(inputs=b0.input, outputs=x)
    newmodel.save("models/dropout_effnetb0.h5")

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
    new_model.save("models/dropout_model.h5")
    return 0

def GradCam(input_model, image, category_index, layer_name, raw_array, dimension, annot):
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
    cam = resize_normalize_cam(cam, dimension)
    
    ovl = calculate_overlap(annot, cam)

    heatmap = cam.copy()

    image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    cam = cv2.applyColorMap(np.uint8(255-255*cam), cv2.COLORMAP_JET)
    cam = cv2.addWeighted(cam, 0.5, image, 0.5, 0)

    return np.uint8(cam), heatmap, ovl


def ScoreCam(input_model, image, category_index, layer_name, raw_array, dimension, annot):
    # Implementation reference
    # https://github.com/tabayashi0117/Score-CAM
    # https://github.com/haofanwang/Score-CAM
    input_shape = (dimension, dimension)
    act_map_array = tf.keras.Model(inputs=input_model.input, outputs=input_model.get_layer(layer_name).output)(image).numpy()
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape, interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(image)
        for k in range(3):
            masked_input[0,:,:,k] = np.multiply(masked_input[0,:,:,k], act_map_normalized, casting="unsafe")
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)

    # No memory available
    input_split = np.array_split(masked_input_array, 40)
    split_weights = []
    for batch in input_split:
        w = input_model(batch)
        split_weights.append(w)
    pred_from_masked_input_array = np.vstack(split_weights)
    weights = pred_from_masked_input_array[:, category_index]
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = resize_normalize_cam(cam, dimension)
    score_heatmap = cam.copy()

    ovl = calculate_overlap(annot, cam)

    raw_image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    cam = cv2.applyColorMap(np.uint8(255-255*cam), cv2.COLORMAP_JET)
    cam = cv2.addWeighted(cam, 0.5, raw_image, 0.5, 0)

    return np.uint8(cam), score_heatmap, ovl

def GradCam_Dropout(input_model, image, category_index, layer_name, raw_array, dimension, sample, annot):
    gradModel = tf.keras.Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(layer_name).output, input_model.output]
        )
    cams = []
    for i in tqdm(range(0, sample)):
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

        cam = resize_normalize_cam(cam, dimension)
        cams.append(cam)

    cams = np.asarray(cams)
    m = np.mean(cams, axis=0)
    no_zero_ratio = len(m[m!=0]) / (dimension*dimension)

    std = np.std(cams, axis=0)
    cov = std / (m + 1e-10)
    cov = cov / np.max(cov) if cov.sum()!=0 else cov.copy()

    w_cam = m * (1-cov)
    w_cam = resize_normalize_cam(w_cam, dimension)
    cov_ovl = calculate_overlap(annot, w_cam)

    heatmap = w_cam.copy()
    stdcam = std / np.max(std) if std.sum()!=0 else std.copy()

    image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    w_cam = cv2.applyColorMap(np.uint8(255-255*w_cam), cv2.COLORMAP_JET)
    w_cam = cv2.addWeighted(w_cam, 0.5, image, 0.5, 0)

    image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    std_cam = cv2.applyColorMap(np.uint8(255-255*stdcam), cv2.COLORMAP_JET)
    std_cam = cv2.addWeighted(std_cam, 0.5, image, 0.5, 0)

    return np.uint8(w_cam), heatmap, np.uint8(std_cam), cov_ovl, no_zero_ratio


def ScoreCam_Dropout(input_model, image, category_index, layer_name, raw_array, dimension, sample, annot):
    # Implementation reference
    # https://github.com/tabayashi0117/Score-CAM
    # https://github.com/haofanwang/Score-CAM
    input_shape = (dimension, dimension)
    cams = []
    for i in tqdm(range(0, sample)):
        act_map_array = tf.keras.Model(inputs=input_model.input, outputs=input_model.get_layer(layer_name).output)(image).numpy()
        act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape, interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)
        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(image)
            for k in range(3):
                masked_input[0,:,:,k] = np.multiply(masked_input[0,:,:,k], act_map_normalized, casting="unsafe")
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)
        
        # No memory available
        input_split = np.array_split(masked_input_array, 40)

        split_weights = []
        for batch in input_split:
            w = input_model(batch)
            split_weights.append(w)
        pred_from_masked_input_array = np.vstack(split_weights)
        weights = pred_from_masked_input_array[:, category_index]
        cam = np.dot(act_map_array[0,:,:,:], weights)
        cam = resize_normalize_cam(cam, dimension)
        cams.append(cam)
    cams = np.asarray(cams)

    m = np.mean(cams, axis=0)
    no_zero_ratio = len(m[m!=0]) / (dimension*dimension)

    std = np.std(cams, axis=0)
    cov = std / (m+1e-10)
    cov = cov / np.max(cov) if cov.sum()!=0 else cov.copy()
    
    newcam = m  * (1-cov)
    newcam = resize_normalize_cam(newcam, dimension)

    cov_ovl = calculate_overlap(annot, newcam)
    score_heatmap = newcam.copy()

    raw_image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    newcam = cv2.applyColorMap(np.uint8(255-255*newcam), cv2.COLORMAP_JET)
    newcam = cv2.addWeighted(newcam, 0.5, raw_image, 0.5, 0)

    stdcam = std / np.max(std) if std.sum()!=0 else std.copy()
    raw_image = np.asarray(Image.fromarray(raw_array.astype("uint8")).resize((dimension, dimension)))
    std_cam = cv2.applyColorMap(np.uint8(255-255*stdcam), cv2.COLORMAP_JET)
    std_cam = cv2.addWeighted(std_cam, 0.5, raw_image, 0.5, 0)

    return np.uint8(newcam), score_heatmap, np.uint8(std_cam), cov_ovl, no_zero_ratio
