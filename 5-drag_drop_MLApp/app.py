from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL, DATA

app = Flask(__name__)
Bootstrap(app)

files = UploadSet('files',ALL)
app.config['UPLOAD_FILES_SWAT'] = 'static/uploadstorage'
configure_uploads(app, files)

import os
import datetime
import time

import pandas as pd
import numpy as np

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/datauploads')
def datauploads():
    return render_template('details.html')