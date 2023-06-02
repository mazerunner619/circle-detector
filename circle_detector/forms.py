from django import forms
from app import models

class ImageUpload(forms.ModelForm):
    class Meta:
        model = models.InputImage
        fields = ['image']