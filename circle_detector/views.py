from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from . import forms

from .utility_functions import circle_detector

import cv2
import numpy as np
import os
import glob

def detect_circle(request):
    if request.method == 'POST':
        image_name = request.GET['img'].split("/")
        img_path = os.path.join(settings.BASE_DIR, image_name[1], image_name[2])
        img = cv2.imread(img_path)
        (bool_res, resultant_image) = circle_detector(img, image_name[2])
        if bool_res == False:
            return HttpResponse('''<h3>'''+resultant_image+'''</h3>''')
        else:
            return render(request, "result.html", {'image_url' : '/media/result-'+image_name[2]})
    else:
        return redirect('homepage')

def homepage(request):
    if request.method == 'POST':
        form = forms.ImageUpload(request.POST, request.FILES)
        print(request.FILES['image'])
        if form.is_valid():
            user_input = form.save()
            image = user_input.image.url
            print(image)
            return render(request, "homepage.html", {'image_url' : image})
    else:
        form = forms.ImageUpload()
        files = glob.glob(os.path.join(settings.BASE_DIR, "media/*"))
        for f in files:
            print("file : ", f)
            if f.endswith(".jpg"):
                os.remove(f)
    return render(request, "homepage.html" , {'form' : form})