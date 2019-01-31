import numpy as np
import os
import json

from io import BytesIO
from server import app
from flask import render_template, request, jsonify

from torchvision.transforms import Compose
from rvl_cdp.data.transforms import Resize, Normalization, ToTensor, PermuteTensor, UnSqueeze
from rvl_cdp.models.factory import get_model
from rvl_cdp.data.factory import get_dataset
from PIL import Image


def read_model(model_path):
    with open(os.path.join(model_path, "model_attrs.json")) as model_attrs_json:
        model_attrs = json.load(model_attrs_json)

    Model = get_model(model_attrs["model_class"])

    fitted_model = Model.load(model_path)
    fitted_model.eval()

    return fitted_model


def classify_image(image):


    with open('server/args.json', 'r') as data_file:
        data = json.load(data_file)
    model_path = data["model_path"]
    dataset_name = data["dataset"]

    Dataset = get_dataset(dataset_name)

    _model = read_model(model_path)

    transforms = Compose([
        Resize((500, 256)),
        Normalization(),
        ToTensor(unsqueeze=True),
        PermuteTensor((2, 0, 1)),
        UnSqueeze()
    ])
    image = transforms({"image": image, "label": None})

    prediction = _model.predict(image["image"])

    return Dataset.get_label_name(int(prediction.data[0]))


@app.route("/classify", methods=["POST"])
def classify():
    request_json = request.json()

    if "image" not in request_json:
        return jsonify({
            "error": "missing image"
        }), 400
    image = request_json["image"]
    prediction = classify_image(image)

    return jsonify({
        "prediction": prediction
    }), 200


@app.route('/upload')
def upload_file_view():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image_bytes = BytesIO(file.stream.read())
        # image = np.fromstring(image, np.uint8)
        img = Image.open(image_bytes)

        bands = img.getbands()
        if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
            img = img.convert('1')

        # produces a PIL Image object
        img = np.asarray(img)
        prediction = classify_image(img)

        return "Class: {}".format(prediction)
