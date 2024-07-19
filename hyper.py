import sys
import os
import glob
import time
from PIL import Image
from enum import Enum
import numpy as np
import absl.logging

class CaptchaType(Enum):
    SUPREME_COURT = "supreme_court"
    GOV24 = "gov24"
    NH_WEB_MAIL = "nh_web_mail"

class Hyper:
    
    def __init__(self, captcha_type:CaptchaType, weights_only=True, quiet_out=False):
        self.NULL_OUT = open(os.devnull, 'w')
        self.STD_OUT = sys.stdout
        self.quiet(quiet_out)
        self.captcha_type = captcha_type
        self.weights_only = weights_only
        self.quiet_out = quiet_out
        self.base_dir = self.get_base_dir()
        self.weights_path = self.get_weights_path(captcha_type, weights_only)
        self.image_width, self.image_height, self.max_length, self.characters, self.labels, self.train_img_path_list = self.get_train_info()

        # Mapping characters to integers
        self.char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.characters, num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def quiet(self, value:bool):

        if value:
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            absl.logging.set_verbosity(absl.logging.ERROR)
            sys.stdout = self.NULL_OUT
        else:
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            tf.get_logger().setLevel('INFO')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
            absl.logging.set_verbosity(absl.logging.INFO)
            sys.stdout = self.STD_OUT

    def get_base_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

    def get_image_files(self, captcha_type:CaptchaType, train=True):
        baseDir = self.get_base_dir()
        imgDir = os.path.join(baseDir, "images", captcha_type.value, "train" if train else "pred")
        return glob.glob(imgDir + os.sep + "*.png")

    def get_train_info(self):
        images = sorted(self.get_image_files(self.captcha_type, train=True))
        image = Image.open(images[-1])
        image_width = image.width
        image_height = image.height
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        return image_width, image_height, max_length, characters, labels, images

    def get_weights_path(self, captcha_type:CaptchaType = None, weights_only:bool = None):
        if captcha_type is None:
            captcha_type = self.captcha_type

        if weights_only is None:
            weights_only = self.weights_only

        weights_path = os.path.join(self.base_dir, "model", captcha_type.value)
        if weights_only:
            weights_path = weights_path + ".weights.h5"

        return weights_path

    def split_dataset(self, batch_size=16, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        image_width, image_height, max_length, characters, labels, train_img_path_list = self.get_train_info()
        images = np.array(train_img_path_list)
        labels = np.array(labels)
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return train_dataset, validation_dataset

    def setBG(self, image_path, color=(255,255,255)):
        img = Image.open(image_path)
        fill_color = color
        img = img.convert("RGBA")
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1]) # omit transparency
            img = background
        image_path = "./temp_white_bg.png"
        img.save(image_path)
        return image_path

    def encode_single_sample(self, image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.image_height, self.image_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": img, "label": label}

    def encode_single_sample_from_bytes(self, image_bytes):
        img = tf.io.decode_image(image_bytes, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.image_height, self.image_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return {"image": img}

    def build_model(self):
        # Inputs to the model
        input_img = layers.Input(
            shape=(self.image_width, self.image_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.image_width // 4), (self.image_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(len(self.characters) + 1, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
   
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res+1)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def load_prediction_model(self):

        if self.weights_only:
            model = self.build_model()
            model.load_weights(self.weights_path)
        else:
            model = keras.models.load_model(self.weights_path)

        sum = model.summary()
        print(sum)

        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
        return prediction_model

    def predict(self, image_path:str, prediction_model=None):

        target_img = self.encode_single_sample(image_path, "")['image']
        target_img = tf.reshape(target_img, shape=[1,self.image_width,self.image_height,1])

        if prediction_model is None:
            prediction_model = self.load_prediction_model()

        pred_val = prediction_model.predict(target_img)
        pred = self.decode_batch_predictions(pred_val)[0]

        return pred

    def predict_from_bytes(self, image_bytes:bytes, prediction_model=None):

        target_img = self.encode_single_sample_from_bytes(image_bytes)['image']
        target_img = tf.expand_dims(target_img, 0)

        if prediction_model is None:
            prediction_model = self.load_prediction_model()

        pred_val = prediction_model.predict(target_img)
        pred = self.decode_batch_predictions(pred_val)[0]

        return pred

    def train_model(self, epochs=100, earlystopping=True, early_stopping_patience:int=8, save_weights:bool=True, save_model:bool=False):
        train_dataset, validation_dataset = self.split_dataset(batch_size=16, train_size=0.9, shuffle=True)
        model = self.build_model()
        
        if earlystopping == True:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            )
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[early_stopping],
            )
        else:
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs
            )

        if save_weights:
            weights_path = self.get_weights_path(self.captcha_type, True)
            model.save(weights_path, save_format='h5')

        if save_model:
            weights_path = self.get_weights_path(self.captcha_type, False)
            model.save(weights_path, save_format='tf')

    def validate_model(self):
        start = time.time()
        matched = 0
        prediction_model = self.load_prediction_model()
        pred_img_path_list = self.get_image_files(self.captcha_type, train=False)

        for pred_img_path in pred_img_path_list:
            pred = self.predict(pred_img_path, prediction_model=prediction_model)
            ori = pred_img_path.split(os.path.sep)[-1].split(".")[0]
            msg = ""
            if(ori == pred):
                matched += 1
            else:
                msg = " Not matched!"
            print("ori : ", ori, "pred : ", pred, msg)

        end = time.time()
        print("Matched:", matched, ", Tottal : ", len(pred_img_path_list), ", Accuracy : ", matched/len(pred_img_path_list) * 100, "%")
        print("pred time : ", end - start, "sec")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_labels = tf.cast(
        ctc_label_dense_to_sparse(y_true, label_length), tf.int32
    )

    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon()
    )

    return tf.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )

def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return tf.expand_dims(tf.range(tf.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = tf.cast(tf.fill([1, label_shape[1]], 0), tf.bool)
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(
        tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(
        tf.reshape(
            tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
            keras.backend.reverse(label_shape, 0),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(
        tf.reshape(keras.backend.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        tf.cast(indices, tf.int64), vals_sparse, tf.cast(label_shape, tf.int64)
    )

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
