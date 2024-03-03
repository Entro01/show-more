import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from flask import session, redirect, url_for
from flask import jsonify

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))


model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

import random

def get_random_images():
    return random.choices(filenames, k=4)

from flask import Flask, url_for, render_template, request

app = Flask(__name__)
#app._static_folder = os.path.join(os.getcwd(), 'show-more/static')
app.secret_key = 'your_secret_key_here'

@app.route('/store_image_path', methods=['POST'])
def store_image_path():
    img_path = request.form.get('img_path')
    session['selected_image_path'] = img_path
    return jsonify({'success': True})

@app.route('/')
def landing_page():
    random_images = get_random_images()
    return render_template('landing.html', images=random_images)

@app.route('/recommend', methods=['GET'])
def recommend_page():
    img_path = session.get('selected_image_path', default='static/images/33634.jpg')
    print(f"Retrieved img_path: {img_path}") # Debugging line
    if img_path is None or not os.path.exists(img_path):
        print("Invalid img_path")
    # Handle the error, e.g., by redirecting to an error page or setting a default image path
    else:
        user_features = feature_extraction(img_path, model)
    user_features = feature_extraction(img_path, model)
    recommended_indices = recommend(user_features, feature_list)
    recommended_paths = [filenames[i] for i in recommended_indices[0]]
    return render_template('recommend.html', images=recommended_paths)

if __name__ == "__main__":
    app.run(debug=True)