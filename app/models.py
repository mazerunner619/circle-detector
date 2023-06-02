from django.db import models
from datetime import datetime

import os

def renamed_file(instance, filename):
    filename = datetime.now().strftime("%d%m%Y%H%M%S%f-") + filename
    upload_to = ""
    return os.path.join(upload_to, filename)

class InputImage(models.Model):
    image = models.ImageField(blank=False, upload_to=renamed_file)