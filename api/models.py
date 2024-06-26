from django.db import models


class User(models.Model):
    username = models.CharField(max_length=20)
    password = models.CharField(max_length=20)


# for storing details of user given for prescription
class UserInputs(models.Model):
    username = models.CharField(max_length=20)
    blood_group = models.CharField(max_length=10)
    work_condition = models.CharField(max_length=20)
    city = models.CharField(max_length=20)
    age = models.IntegerField()
    image = models.ImageField(upload_to='user_inputs')
    processed_image = models.TextField(null=True)  # Store the processed image as base64
    df_data = models.TextField(null=True)  # Store the DataFrame as JSON
    date = models.DateField
    time = models.TimeField
    disease = models.JSONField(blank=True, null=True)
    probability = models.JSONField(blank=True, null=True)