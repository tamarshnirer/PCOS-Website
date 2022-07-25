from django.shortcuts import render
from django.core.files import File
from django.http import HttpResponse
import numpy as np
import pickle
import pandas as pd
from sklearn.base import TransformerMixin
import os
from django.conf import settings
from sklearn.linear_model import LogisticRegression



def home(request):
    return render(request, 'site/index.html')

def aboutus(request):
    return render(request, 'site/aboutus.html')

def aboutpcos(request):
    return render(request, 'site/aboutPCO.html')

def form(request):
    return render(request, 'site/PCOtest.html')


def result(request):
    vector = []
    shannon_genefamilies_cpm = request.GET.get('shanon_genefamilies_cpm')
    vector.append(shannon_genefamilies_cpm)
    bmi = request.GET.get('bmi')
    vector.append(bmi)
    vector = np.array(vector)
    vector = vector.astype(dtype='float64')
    vector_df = pd.DataFrame([vector], columns=['shanon_genefamilies_cpm', 'bmi'])
    with open('C:\\Users\\PC\\Desktop\\PCOS_website-project\\models\\scaler2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    vector_df = scaler.transform(vector_df)
    with open('C:\\Users\\PC\\Desktop\\PCOS_website-project\\models\\model2.pkl', 'rb') as f:
        clf = pickle.load(f)
    prob = (clf.predict_proba(vector_df)[0][1])*100
    pow10 = 10 ** 2
    prob = prob * pow10 // 1 / pow10
    return render(request, 'site/result.html', {'prob': prob})

def disclaimer(request):
    return render(request, 'site/Disclaimer.html')
