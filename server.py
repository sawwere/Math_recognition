from flask import Flask, jsonify, request, flash, redirect, url_for, send_from_directory, g
from flask import abort
from flask import make_response
from flask import render_template
import sqlite3

import jsonpickle

from werkzeug.utils import secure_filename

import os
import time
from utils.Node import *
from utils.image_processing import *
from utils.image_info import ImageInfo
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
        MODEL_PATH = 'models\mathnet224\mathnet8.ml',
        INPUT_SIZE=(224, 224),
        VISUALIZE=False,
        MIN_CONF=0.05,
        DEBUG=False
    )

def get_db():
    """ Возвращает объект соединения с БД"""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Закрывает соединение с с БД"""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()



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
def get_submit_page():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history_page():
    return render_template('history.html')

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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return proccess_file(filename, file)

@app.route('/', methods=['POST'])
def load_image():
    return redirect(url_for('get_submit_page'), code=307)

def proccess_file(filename, file):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(path)
    image_info = ImageInfo(image)
    css = ContourSearchStep(app.config['kwargs'] )
    info1 = css.process(image_info)
    result_name = 'result_'+str(time.time())+'_'+filename
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], result_name), info1.image)

    groups_step = GroupSplittingStep()
    info2 = groups_step.process(info1)
    tree_step = BuildTreeStep()
    info3 = tree_step.process(info2)
    return render_template('submit.html', 
                           orig_filename=filename,
                           result_filename=result_name, 
                           json=jsonpickle.encode(info3.nodes, indent=2))


if __name__ == '__main__':
    init_db()
    app.run(debug=True)