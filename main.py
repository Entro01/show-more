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
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

import random

def get_random_images():
    return random.choices(filenames, k=4)

from flask import Flask, url_for, render_template, request

app = Flask(__name__)

@app.route('/')
def landing_page():
    random_images = get_random_images()
    print(random_images)
    return render_template('landing.html', images=random_images)

#@app.route('/recommend', methods=['POST'])
#def recommend_page():
#    user_image = request.files['image']
#    user_image.save("temp.jpg")
#    user_features = feature_extraction("temp.jpg", model)
#    recommended_images = recommend(user_features, feature_list)
#    image_urls = [url_for('static', filename=img) for img in recommended_images]
#    return render_template('recommend.html', images=image_urls)

if __name__ == "__main__":
    app.run(debug=True)
