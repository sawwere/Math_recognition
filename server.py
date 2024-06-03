from flask import Flask, jsonify, request, flash, redirect, url_for, send_from_directory, g
from flask import abort
from flask import make_response
from flask import render_template
import sqlite3

import jsonpickle

from werkzeug.utils import secure_filename

import os
import time
from datetime import datetime
from pathlib import Path

from utils.Node import *
from utils.image_processing import *
from utils.image_info import ImageInfo
from utils.steps.lexer_step import *
from utils.steps.image_processing_step import *

RESOURCE_FOLDER = 'resources'
UPLOAD_FOLDER = RESOURCE_FOLDER + '/uploads'
RESULT_FOLDER = RESOURCE_FOLDER + '/results'
ALLOWED_EXTENSIONS = {'bmp',  'png', 'jpg', 'jpeg'}
DATABASE = 'table.db'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['kwargs'] = dict(
        MODEL_PATH = 'models\mathnet224\mathnet.ml',
        MODEL_KIND = 'RES_NET',
        INPUT_SIZE=(224, 224),
        VISUALIZE=False,
        MIN_CONF=0.15,
        LEXER_MULTIPLIER = 1.5,
        DEBUG=False
    )



def allowed_file(filename):
    """ Функция проверки расширения файла """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/resources/<path:filename>')
def download_file(filename):
    return send_from_directory(RESOURCE_FOLDER, filename, as_attachment=False)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def get_exact_submit():
    filename = request.args.get('name')
    if filename is None:
        return render_template('index.html')
    orig = str(filename)
    path_to_dir = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(path_to_dir):
        return not_found(request.url)
    with open(path_to_dir+'/result.json', 'r') as file:
        json_string = file.read()
    return render_template('submit.html', 
                           orig_filename=orig,
                           result_dir=filename,
                           json=json_string)

@app.route('/history', methods=['GET'])
def get_history_page():
    dirs = [x[1] for x in os.walk(app.config['RESULT_FOLDER'])][0]
    times = [datetime.fromtimestamp(float(dir.split('.')[0])) for dir in dirs]
    return render_template('history.html', total=len(dirs), results=dirs, times=times)

@app.route('/submit', methods=['POST'])
def get_submit():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Нет выбранного файла')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = str(time.time()) + '_' + filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return proccess_file(filename, request.form)

@app.route('/', methods=['POST'])
def load_image():
    return redirect(url_for('get_submit'), code=307)

def proccess_file(filename, form):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(path)
    image_info = ImageInfo(image)

    kwargs = app.config['kwargs'].copy()
    if form.get('options') is not None:
        if form.get('options') == 'AlexNet':
            kwargs['MODEL_KIND'] = 'ALEX_NET'
            kwargs['MODEL_PATH'] = 'models\\alexnet227\mathnet.ml'
            kwargs['INPUT_SIZE']=(227, 227)
            model = models.alexnet.AlexNet(mnt.NUM_CLASSES)
        else:
            kwargs['MODEL_KIND'] = 'RES_NET'
            kwargs['MODEL_PATH'] = 'models\\mathnet224\mathnet.ml'
            model = mnt.MathNet()
    
    model.load_state_dict(torch.load(kwargs['MODEL_PATH']))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    css = ContourSearchStep(kwargs, model)
    info1 = css.process(image_info)
    path_to_dir = os.path.join(app.config['RESULT_FOLDER'], filename)
    Path(path_to_dir).mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(os.path.join(path_to_dir, 'result.jpg'), info1.image)

    groups_step = GroupSplittingStep(kwargs)
    info2 = groups_step.process(info1)
    if form.get('flexSwitchCheck_useLexer') is not None:
        lexer_step = LexerStep(kwargs)
        info2 = lexer_step.process(info2)
    tree_step = BuildTreeStep(kwargs)
    info3 = tree_step.process(info2)

    json_string = jsonpickle.encode(info3.nodes, indent=2)
    with open(path_to_dir+'/result.json', 'w') as file:
        file.write(json_string)
    return render_template('submit.html', 
                           orig_filename=filename,
                           result_dir=filename,
                           json=json_string)


if __name__ == '__main__':
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['RESULT_FOLDER']).mkdir(parents=True, exist_ok=True)
    app.run(debug=True)