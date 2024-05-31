import os
basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

RESOURCE_FOLDER = 'resources'
UPLOAD_FOLDER = RESOURCE_FOLDER + '/uploads'
RESULT_FOLDER = RESOURCE_FOLDER + '/results'
ALLOWED_EXTENSIONS = {'bmp',  'png', 'jpg', 'jpeg'}

CONTOURS_DETECTOR_KWARGS = dict(
        MODEL_PATH = 'models\mathnet224\mathnet8.ml',
        INPUT_SIZE=(224, 224),
        VISUALIZE=False,
        MIN_CONF=0.05,
        DEBUG=False
    )