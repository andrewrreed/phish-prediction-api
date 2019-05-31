# serve Phish prediction model as a Flask application

import os
import util
import numpy as np
from keras.models import load_model
from flask import Flask, request

app = Flask(__name__)
model = None
idx_to_song = None

#------------------ Load Model and Encodings ------------------ 
def load_models():
    global model
    global idx_to_song

    model = load_model('models/model.nn_arch_2-150-seqlen-100-lstmunits-0.5-b_dropout-0.5-a_dropout.hdf5')
    model._make_predict_function()
    idx_to_song = util.load_pickle_object('models/idx_to_song.pkl')

    print('Successfully loaded the model.')
    return None

#------------------ Create API's ------------------ 
@app.route('/')
def home_endpoint():
    return ('Take care of your shoes!')

@app.route('/next-song', methods=['POST'])
def predict_next_song():
    # takes in a numpy array of length 150 representing encoded songs
    if request.method == 'POST':
        data = request.get_json()
        data_formatted = np.array([data]) # converts shape from (150,) --> (1,150)
        prediction = model.predict_classes(data_formatted)
        prediction_clean = idx_to_song[prediction.item()] # un-encode the prediction

    return prediction_clean
