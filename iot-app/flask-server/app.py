import base64
import random
import shutil
import string
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np
from skimage import transform
import cv2

app = Flask(__name__)
CORS(app)
class_labels = {'angry cat': 0,
                'annoyed cat': 1,
                'excited cat': 2,
                'friendly cat': 3,
                'happy cat': 4,
                'relaxed cat': 5,
                'scared cat': 6,
                'sleepy cat': 7}
# Recreate the exact same model, including its weights and the optimizer
cat_model = tf.keras.models.load_model('vgg-iot.h5')

alphabet = string.ascii_lowercase + string.digits


def random_choice():
    return ''.join(random.choices(alphabet, k=4))


# cat_model.summary()
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def scandirs(path):
    for root, dirs, files in os.walk(path):
        for currentFile in files:
            print("processing file: " + currentFile)
            exts = ('.jpg', '.webp')
            if currentFile.lower().endswith(exts):
                os.remove(os.path.join(root, currentFile))


@app.route('/')
def hello_world():
    return 'Hello Worlssd!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    filename = uuid.uuid4()
    photo = request.get_json()['user_photo']

    photo_data = base64.b64decode(photo)

    with open(f"{filename}.jpg", "wb") as file:
        file.write(photo_data)
    image = load(f"{filename}.jpg")
    preds = cat_model.predict(image)
    y_classes = preds.argmax(axis=-1)
    cat_expression = list(class_labels.keys())[list(class_labels.values()).index(y_classes)]
    print(cat_expression)
    return cat_expression


def makedirs(train_path):
    path_list = ['angry cat', 'annoyed cat','excited cat','friendly cat','happy cat','relaxed cat','scared cat','sleepy cat']
    for path in path_list:
        new_dir_path = os.path.join(train_path, path)
        os.makedirs(new_dir_path)


def deletedirs(train_path):
    shutil.rmtree(train_path)


@app.route('/train_image', methods=['POST'])
def train_image():
    random_string = random_choice()
    train_path = r"train/" + random_string
    # make dirs from train path
    makedirs(train_path)
    scandirs(train_path)

    filename = uuid.uuid4()
    photo = request.get_json()['user_photo']
    label = request.get_json()['label']
    photo_data = base64.b64decode(photo)
    cat_expression = list(class_labels.keys())[list(class_labels.values()).index(label)]
    print(f"train/{random_string}/{cat_expression}/{filename}.jpg")
    with open(f"train/{random_string}/{cat_expression}/{filename}.jpg", "wb") as file:
        file.write(photo_data)
    img_arr = cv2.imread(f"train/{random_string}/{cat_expression}/{filename}.jpg")

    img_arr = cv2.resize(img_arr, (224, 224))

    x_train = [img_arr]
    train_x = np.array(x_train)
    train_x = train_x / 255.0

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='sparse')
    val_set = train_datagen.flow_from_directory(train_path,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='sparse')
    train_y = training_set.classes
    print(train_y)
    val_y = val_set.classes
    # tell the model what cost and optimization method to use
    opt = keras.optimizers.Adam(learning_rate=0.001)
    cat_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    cat_model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=2)
    cat_model.save("vgg-iot.h5")
    deletedirs(train_path)
    return 'model updated'


if __name__ == '__main__':
    app.run(port=5000)
