import io
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from patchify import patchify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tqdm import tqdm

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def build_model(num_classes):
    IMG_SIZE = 224
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Load the model once at startup
NUM_CLASSES = 3
MODEL_PATH = "Efficient_Net_Final.weights.h5"
model = build_model(num_classes=NUM_CLASSES)
model.load_weights(MODEL_PATH)
class_labels = ['benign', 'malignant', 'normal']


@app.route('/predict_video_dl', methods=['POST'])
def predict_video_dl():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output_video_path = process_video_dl(file_path)

        return send_file(output_video_path, as_attachment=True)

    return jsonify({'error': 'File type not allowed'}), 400


def process_video_dl(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_filename = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(video_path))
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for prediction
        image = cv2.resize(frame, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Predict class labels
        preds = model.predict(image)
        max_index = np.argmax(preds[0])
        predicted_class = class_labels[max_index]

        # Annotate frame with prediction
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2,
                    cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    return output_filename


@app.route('/predict_video_llm', methods=['POST'])
def predict_video_llm():
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output_video_path = process_video_llm(file_path)

        return send_file(output_video_path, as_attachment=True)

    return jsonify({'error': 'File type not allowed'}), 400


def process_video_llm(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model state dict
    path = 'preTrainedVit_50_epochs_.pth'
    pretrained_vit_state_dict = torch.load(path, map_location=device)

    # Setup a ViT model instance with pretrained weights
    pretrained_vit = models.vit_b_16().to(device)

    # Rename the keys in the loaded state dict to match the keys in the model's state dict
    new_pretrained_vit_state_dict = {}
    for k, v in pretrained_vit_state_dict.items():
        if 'head' in k:
            k = k.replace('head', 'heads')  # Rename keys related to the classifier head
        new_pretrained_vit_state_dict[k] = v

    # Load the modified state dict into the model
    pretrained_vit.load_state_dict(new_pretrained_vit_state_dict, strict=False)

    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # Change the classifier head
    class_names = ['benign', 'malignant', 'normal']
    pretrained_vit.conv_proj.in_channels = 1
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    # Define the transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(video_path))
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for prediction
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_image = transform(image).unsqueeze(0).to(device)

        # Predict class labels
        pretrained_vit.eval()
        with torch.no_grad():
            outputs = pretrained_vit(input_image)

        # Get predicted class index
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

        # Annotate frame with prediction
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

    return output_filename


def annotate_image(image, predicted_class):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Add text to the image
    text = f'{predicted_class}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (255, 0, 0)  # White color
    line_type = 2
    text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
    text_x = int((image_cv.shape[1] - text_size[0]) / 2)  # Centered horizontally
    text_y = text_size[1] + 20  # 20 pixels below the top
    cv2.putText(image_cv, text, (text_x, text_y), font, font_scale, font_color, lineType=line_type, thickness=4)

    return image_cv


@app.route('/predict_image_llm', methods=['POST'])
def predict_image_llm():
    try:
        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the saved model state dict
        path = 'preTrainedVit_50_epochs_.pth'
        pretrained_vit_state_dict = torch.load(path, map_location=device)

        # Setup a ViT model instance with pretrained weights
        pretrained_vit = models.vit_b_16().to(device)

        # Rename the keys in the loaded state dict to match the keys in the model's state dict
        new_pretrained_vit_state_dict = {}
        for k, v in pretrained_vit_state_dict.items():
            if 'head' in k:
                k = k.replace('head', 'heads')  # Rename keys related to the classifier head
            new_pretrained_vit_state_dict[k] = v

        # Load the modified state dict into the model
        pretrained_vit.load_state_dict(new_pretrained_vit_state_dict, strict=False)

        # Freeze the base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False

        # Change the classifier head
        class_names = ['benign', 'malignant', 'normal']
        pretrained_vit.conv_proj.in_channels = 1
        pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

        # Define the transformations for the input image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Read the image file from request data
        image_data = request.files['image']

        # Convert binary data to PIL Image
        image = Image.open(image_data).convert('RGBA')  # Convert PNG to RGBA format

        # Preprocess the image
        input_image = transform(image).unsqueeze(0).to(device)

        # Set the model to evaluation mode
        pretrained_vit.eval()

        # Make prediction
        with torch.no_grad():
            outputs = pretrained_vit(input_image)

        # Get predicted class index
        _, predicted = torch.max(outputs, 1)

        # Map the predicted index to class label
        predicted_class = class_names[predicted.item()]

        # Annotate the image with predicted class
        annotated_image = annotate_image(image, predicted_class)

        # Convert annotated image back to PIL format
        annotated_image_pil = Image.fromarray(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGBA))  # Convert BGR to RGBA for PIL

        # Save the annotated image to a temporary file or memory buffer (BytesIO)
        output_buffer = io.BytesIO()
        annotated_image_pil.save(output_buffer, format='PNG')  # Save as PNG format
        output_buffer.seek(0)

        # Return the annotated image as response
        return send_file(output_buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)})


# Predict using DL model
@app.route('/predict_image_dl', methods=['POST'])
def predict_image_dl():
    try:
        # Read the image file from request data
        image_data = request.files['image']

        # Convert binary data to PIL Image
        image = Image.open(image_data).resize((224, 224))

        path = "Efficient_Net_Final.weights.h5"

        NUM_CLASSES = 3

        model = build_model(num_classes=NUM_CLASSES)

        model.load_weights(path)

        x = np.expand_dims(image, axis=0)
        x = preprocess_input(x)

        # Predict class labels
        preds = model.predict(x)
        class_labels = ['benign', 'malignant', 'normal']
        max_index = np.argmax(preds[0])
        predicted_class = class_labels[max_index]

        # Annotate the image with predicted class
        annotated_image = annotate_image(image, predicted_class)

        # Convert annotated image back to PIL format
        annotated_image_pil = Image.fromarray(
            cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGBA))  # Convert BGR to RGBA for PIL

        # Save the annotated image to a temporary file or memory buffer (BytesIO)
        output_buffer = io.BytesIO()
        annotated_image_pil.save(output_buffer, format='PNG')  # Save as PNG format
        output_buffer.seek(0)
        return send_file(output_buffer, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)})

# Define Dice Coefficient and Dice Loss
smooth = 1e-15

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load the model
model_path = r"model256_95.5_acc.keras"  # Change to your model path
model1 = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

@app.route('/predict_image', methods=['POST'])
def predict():
    # UNETR Configuration
    cf = {}
    cf["image_size"] = 256
    cf["num_channels"] = 3
    cf["num_layers"] = 12
    cf["hidden_dim"] = 64
    cf["mlp_dim"] = 32
    cf["num_heads"] = 6
    cf["dropout_rate"] = 0.1
    cf["patch_size"] = 16
    cf["num_patches"] = (cf["image_size"] // cf["patch_size"]) ** 2
    cf["flat_patches_shape"] = (
        cf["num_patches"],
        cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
    )
    # Get image and mask from request
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'No image or mask provided'}), 400

    image_file = request.files['image']
    mask_file = request.files['mask']

    # Convert image and mask to numpy arrays
    image = Image.open(image_file).convert('RGB')
    mask = Image.open(mask_file).convert('L')

    image = np.array(image)
    mask = np.array(mask)

    # Preprocess image
    image_resized = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    x = image_resized / 255.0

    # Patchify image
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, (cf["patch_size"], cf["patch_size"], cf["num_channels"]), cf["patch_size"])
    patches = patches.reshape(-1, cf["patch_size"] * cf["patch_size"] * cf["num_channels"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    # Preprocess mask
    mask_resized = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask_normalized = mask_resized / 255.0

    # Prediction
    pred = model1.predict(patches, verbose=0)[0]
    pred_resized = cv2.resize(pred, (cf["image_size"], cf["image_size"]))

    # Convert predicted mask to RGB
    pred_colored_resized = np.stack([pred_resized]*3, axis=-1) * 255  # Stack along channel axis

    # Create overlay
    opacity = 0.5
    overlay = (image_resized * (1 - opacity) + pred_colored_resized * opacity).astype(np.uint8)

    # Preparing actual mask for comparison
    mask_colored = np.stack([mask_resized]*3, axis=-1).astype(np.uint8)

    # Save final result
    line = np.ones((image_resized.shape[0], 10, 3)) * 255
    cat_images = np.concatenate([overlay, line, mask_colored], axis=1)

    # Convert to PIL Image for sending response
    result_image = Image.fromarray(cat_images.astype('uint8'), 'RGB')
    img_io = io.BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5000, debug=True)
