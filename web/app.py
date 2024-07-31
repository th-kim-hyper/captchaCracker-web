import sys, time, os
from os import path

BASE_DIR = path.dirname( path.dirname( path.abspath(__file__) ) )
sys.path.append(path.dirname(path.dirname(path.abspath(__file__) ) ))

from flask import Flask
from flask import render_template, flash, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from hyper import CaptchaType, Hyper
from PIL import Image

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['BASE_DIR'] = BASE_DIR
app.config['IMAGE_DIR'] = IMAGE_DIR 
app.config['MODEL_DIR'] = MODEL_DIR
app.config['UPLOAD_DIR'] = UPLOAD_DIR

models = {}
models[CaptchaType.SUPREME_COURT.value] = Hyper(captcha_type=CaptchaType.SUPREME_COURT, weights_only=True, quiet_out=False)
models[CaptchaType.GOV24.value] = Hyper(captcha_type=CaptchaType.GOV24, weights_only=True, quiet_out=False)

def predict(model_type, file):
    
    start_time = time.time()
    model = models[model_type]
    filename = f"{int(time.time())}.png"
    save_path = os.path.join(IMAGE_DIR, model_type, "uploads", filename)

    with Image.open(file.stream) as image:
        image = Image.open(file.stream)
        image = setBGColor(image)
        image.save(save_path)  

    pred = model.predict(save_path)
    file_name = save_path.split(os.sep)[-1]
    
    p_time = time.time() - start_time
    
    result = {}
    result['model_type'] = model_type
    result['save_path'] = save_path
    result['file_name'] = file_name
    result['pred'] = pred
    result['p_time'] = p_time
    
    return result

def setBGColor(image, fill_color = (255,255,255)):
    color_bg = Image.new("RGBA", image.size, fill_color)
    color_bg.paste(image, (0, 0), image)
    color_bg = color_bg.convert("L")
    image.close()
    return color_bg

@app.route('/', methods=['GET', 'POST'])
def index(name=None):

    result = {}

    if request.method == 'POST':
        file = request.files['captchaFile']
        
        if(file != None):
            model_type = request.form['modelType'] if request.form['modelType'] is None else CaptchaType.SUPREME_COURT.value
            result = predict(model_type, file)

    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def predictApi(name=None):

    result = {}

    if request.method == 'POST':
        file = request.files['captchaFile']
        
        if(file != None):
            model_type = request.form['modelType'] if request.form['modelType'] is None else CaptchaType.SUPREME_COURT.value
            result = predict(model_type, file)
        
    return jsonify(result)

@app.route('/images', methods=['GET'])
def images(name=None):
    model_type = request.args.get('t')
    file_name = request.args.get('f')
    filepath = os.path.join(IMAGE_DIR, model_type, "uploads", file_name)
    return send_file(filepath, mimetype='image/png')

if __name__ == '__main__':
    app.debug = False
    app.run("0.0.0.0")