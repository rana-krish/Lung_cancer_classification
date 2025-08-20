from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from PIL import Image
import pickle
import os

# Load the model components
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
pca = pickle.load(open(os.path.join(BASE_DIR, 'pca.pkl'), 'rb'))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))

def home(request):
    return render(request, 'classifier/index.html')

def predict(request):
    if request.method == 'POST' and 'image' in request.FILES:
        img_file = request.FILES['image']
        try:
            img = Image.open(img_file).convert('RGB').resize((64, 64))
            img_array = np.array(img).flatten().astype(np.float32)
            img_array = scaler.transform([img_array])
            img_array = pca.transform(img_array)
            pred = model.predict(img_array)[0]
            classes = {0: 'adenocarcinoma', 1: 'benign', 2: 'squamous_carcinoma'}
            result = classes[pred]
            return HttpResponse(result)
        except Exception as e:
            return HttpResponse(f"Error: {str(e)}")
    return HttpResponse("Invalid request.")