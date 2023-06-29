from django.conf import settings
from django.db import models
import os


# def get_image(instance, filename):
#
#     ext = filename.split('.')[-1]
#     filename = "ROC-%s.%s" %(instance.id, ext)
#     return os.path.join(settings.MEDIA_URL, filename)
class ImageModel(models.Model):
    image = models.FileField(max_length=255 , null=True)
    cm = models.FileField(max_length=255, null=True)

    def __str__(self):
        return "ROC " + str(self.id)