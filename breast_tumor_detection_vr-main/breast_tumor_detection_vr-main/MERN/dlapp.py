from flask import Flask, request, send_file
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
import os
import io
app = Flask(__name__)
smooth = 1e-15


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# Load the model with custom objects
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = load_model("model.keras")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        video_file = request.files['video_file']
        video_file.save('temp.mp4')  # Save the file to a temporary location
        video = cv2.VideoCapture('temp.mp4')
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []

        for i in range(total_frames):
            ret, frame = video.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(image)
            if i == total_frames - 1:
                break

        video.release()

        for i in range(len(images)):
            image = images[i]
            image = cv2.resize(image, (224, 224))
            x = image / 255.0
            x = np.expand_dims(x, axis=0)

            y_pred = model.predict(x, verbose=0)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred >= 0.5
            y_pred = y_pred.astype(np.uint8)
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
            y_pred = y_pred * 255

            image = image.astype(np.uint8)
            overlay = cv2.addWeighted(image, 0.5, y_pred, 0.5, 0)
            images[i] = overlay

        output_video = 'predvideo.mp4'
        frame_rate = 1
        width = 640
        height = 480

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

        for i in images:
            img = i
            img = cv2.resize(img, (width, height))
            video_writer.write(img)

        video_writer.release()

        return send_file(output_video, mimetype='video/mp4')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        image_file = request.files['image_file']
        image_file.save('temp.jpg')  # Save the file to a temporary location
        image = cv2.imread('temp.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.uint8)
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255

        image = image.astype(np.uint8)
        overlay = cv2.addWeighted(image, 0.5, y_pred, 0.5, 0)
        overlay = cv2.resize(overlay, (640, 480))
        ret, jpeg = cv2.imencode('.jpg', overlay)
        return send_file(io.BytesIO(jpeg), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
