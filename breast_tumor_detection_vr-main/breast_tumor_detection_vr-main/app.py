from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Directory where uploaded files will be stored
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def save_file(file, filename):
    try:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename)))
    except Exception as e:
        return str(e)
    return f'{filename} uploaded successfully'


@app.route('/upload_video', methods=['POST'])
def upload_video():
    files = {
        'input_video_dl': 'input_video_dl.mp4',
        'input_video_llm': 'input_video_llm.mp4',
        'video_dl_classification': 'video_dl_classification.mp4',
        'video_llm_classification': 'video_llm_classification.mp4',
        'video_dl_segmentation': 'video_dl_segmentation.mp4',
        'video_llm_segmentation': 'video_llm_segmentation.mp4'
    }

    response = {}
    for key, filename in files.items():
        if key in request.files:
            response[key] = save_file(request.files[key], filename)

    return jsonify(response)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    files = {
        'input_image_dl': 'input_image_dl.png',
        'input_image_llm': 'input_image_llm.png',
        'image_dl_classification': 'image_dl_classification.png',
        'image_llm_classification': 'image_llm_classification.png',
        'image_dl_segmentation': 'image_dl_segmentation.png',
        'image_llm_segmentation': 'image_llm_segmentation.png'
    }

    response = {}
    for key, filename in files.items():
        if key in request.files:
            response[key] = save_file(request.files[key], filename)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5050)
