#info on how to start the project, basically make the venv and have all packages installed and then turn on the mongodb using task manager or something, make the server running , open mongodb atlas and open the connection simple
import pymongo
from django.http import HttpResponse, HttpResponseRedirect
import datetime
from django.contrib import messages
from django.shortcuts import render,redirect
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.hashers import make_password,check_password
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
import keras
from django.conf import settings
import os
import numpy as np
import io
import base64
import os
import numpy as np
import os
import numpy as np
import cv2
import keras
import matplotlib
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import tensorflow as tf
# from sorl.thumbnail import ImageField, get_thumbnail
from .google_scipt_py import object_detection


#rendering templates
def Homepage(request):
    return render(request,  'Homepage.html')

def login(request):
    if 'username' not in request.session:
        return render(request,  'login.html')
    return HttpResponseRedirect('/dashboard')

def prediction_form(request):
    return render(request, 'prediction_form.html')


def dashboard(request):
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['TempUser']

    # Check to make sure that the username key is in the request.session dictionary.
    if 'username' not in request.session:
        return render(request, 'Dashboard.html', {'message': 'You must be logged in to view your dashboard.'})

    # Get the username from the request.session dictionary.
    username = request.session['username']

    # Use the MongoDB cursor object to iterate over the documents in the user_inputs collection.
    cursor = db.user_inputs.find({'username': username})

    # Create a list to store the data.
    data = []
    cnt = 1
    # Iterate over the cursor and add each document to the list.
    for document in cursor:
        date = document['date'].get('date')
        time = document['time'].get('time')
        blood_grp = document['blood_group']
        disease1 = document['disease']
        id= document['_id']
        
        # disease2 = document['disease'].get('1')
        
        probability1 = document['probability']

        # probability2 = document['probability'].get('1')
        data.append({
            's_no': cnt,
            'date': date,
            'time': time,
            'blood_grp': blood_grp,
            # 'disease1': disease1,
            # 'disease2': disease2,
            # 'probability1': probability1,
            # 'probability2':probability2,
            'disease1': disease1,
            'probability1': probability1,
            'id':id
            # 'disease2': disease2,
            # 'probability1': probability1,
            #disease 1 is the true or false value now and i cant change variable name cuz butterfly effect now
        })

        cnt += 1

    # Create a context dictionary.
    context = {
        'data': data
    }

    # Render the Dashboard.html template, passing in the context dictionary.
    return render(request, 'Dashboard.html', context)




import cv2


def convert_ndarray_to_list(ndarray):
    data = []
    for i in range(ndarray.shape[0]):
        data.append( ndarray[i]*100)
    # print(data)
    context = {
        'data': data
    }
    return context



def convert_img(img):
    '''
        This function returns the preprocessed and changed dimension of nparray of image passed 
    '''
    img_bytes = io.BytesIO(img.read())
    img = tf.keras.utils.load_img(img_bytes, target_size=(100, 100))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def sort_convert(preds):
    disease = ["Violence"]
    disease=list_to_dict(disease)
    # Get the NumPy array of probabilities
    probabilities = np.array(preds)
    # Sort the probabilities in descending order
    # sorted_probabilities = np.sort(probabilities)[::-1]

    # Get the top 2 probabilities
    # top_2_probabilities = sorted_probabilities[:2]
    # rounded_array = np.around(top_2_probabilities, decimals=2)
    probability= numpy_ndarray_to_dict(probabilities)
    # Get the indices of the top 2 probabilities
    # top_2_indices = np.argsort(probabilities)[::-1][:2]


    # Create a context dictionary
    context = {
        "disease": disease,
        "probability": probability,
    }
    return context

def list_to_dict(list):
    """Converts a list to a dictionary.

    Args:
        list: A list.

    Returns:
        A dictionary.
    """   

    dict = {}
    for i in range(len(list)):
        dict[i] = list[i]
    return dict


def numpy_ndarray_to_dict(ndarray):
  """Converts a NumPy array to a dictionary.

  Args:
    ndarray: A NumPy array.

  Returns:
    A dictionary.
  """

  flattened_array = ndarray.flatten()
  dict = {}
  for i in range(len(flattened_array)):
    dict[i] = np.round(flattened_array[i]*100,2)
  return dict

# @csrf_exempt
# def save_data(request):
#     if request.method == 'POST':
#         blood_group = request.POST['blood_group']
#         work_condition = request.POST['work_condition']
#         city = request.POST['city']
#         age = request.POST['age']
#         img = request.FILES['image']
#         print(img)

#         # Load the ML model.
#         ML_MODELS_DIR = settings.ML_MODELS_DIR
#         import shutil

# #version checker

# # print("numpy version:", np._version_)
# # print("opencv version:", cv2._version_)
# # print("keras version:", keras._version_)
# # print("matplotlib version:", matplotlib._version_)
# # print("pickle version:", pickle.format_version)
#         model1=load_model(os.path.join(ML_MODELS_DIR, 'model.h5'))
#         model2=load_model(os.path.join(ML_MODELS_DIR, 'modelnew.h5'))
#         # model2=load_model('modelnew.h5')

#         trueorfalse=True
#         proba=0
#         def process_image(image_path, model, Q, display=False):
#     # Load the image
#             image = cv2.imread(image_path)
    
#     # Clone the output frame, then convert it from BGR to RGB
#             output = image.copy()
#             frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (128, 128)).astype("float32")
#             frame = frame.reshape(128, 128, 3) / 255

#             # Make predictions on the frame and then update the predictions queue
#             preds = model.predict(np.expand_dims(frame, axis=0))[0]
#             Q.append(preds)

#     # Perform prediction averaging over the current history of previous predictions
#             results = np.array(Q).mean(axis=0)
#             label = results > 0.7
#             probability = results
#             proba=probability
#             trueorfalse=label

#             print(probability)
#             if not label:
#                 text_color = (0, 255, 0) # default : green
#                 text = "Violence:False, True prob={:.2f}".format((1.38-probability[0])/1.38)
#             if label: # Violence prob
#                 text_color = (0, 0, 255) # red

#                 text = "Violence: True, Prob= {:.2f}".format(probability[0])
#             FONT = cv2.FONT_HERSHEY_SIMPLEX
        
#             cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)
        
#             if display:
#                 cv2.imshow('Output', output)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
        
#             return label, output, probability*100
        
#         # def analyze_images(image_paths, model, threshold=2):
#         def analyze_images(image_paths, model, threshold=1):
#             print("Loading model ...")
#             model = model1
#             print(type(image_paths))
#             Q = deque(maxlen=128)
    
#             violence_count = 0
#             output_images = []
#             for image_path in image_paths:
#                 label, output, prob = process_image(image_path, model, Q, display=False)
#                 output_images.append((label, output, prob))
#                 if label:
#                     violence_count += 1
        
#             for idx, (label, output, prob) in enumerate(output_images):
#                 plt.figure()
#                 plt.title("Violence Detected" if label else "No Violence Detected")
#                 plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
#                 plt.axis('off')
#                 plt.show()
#                 print(prob)
        
#             if violence_count >= threshold:
#                 print("Violence detected in {} out of {} images.".format(violence_count, len(image_paths)))
#                 return True
#             else:
#                 print("No violence detected.")
#                 return False
        
#         # Usage example
#         image_paths = [
#             "image_checker_1.png",  # Path to your image files
            
#         ]
        
#         # analyze_images(image_paths, model1)#actual implementation sort of in google colab
#         analyze_images(img, model1)
#         # analyze_images(image_paths, model2)
#         #above is removed due to main error considering 2 models were made when it came from same model
#         # model = keras.models.load_model(os.path.join(ML_MODELS_DIR, 'MLmodel.h5'))
#         # # Convert and preprocess the image file.
#         # x=convert_img(img)
#         # preds = model.predict(x)
#         # context=sort_convert(preds)
#         # print(context)


#         #to store in mongodb
#         client = pymongo.MongoClient('mongodb://localhost:27017')
#         db = client['TempUser']

#         # Get the phone number of the logged in user.
#         username = request.session['username']

#         # Get the current date and time.
#         now = datetime.datetime.now()

#         date = now.date()
#         date_str = date.isoformat()
#         date_dict = dict(date=date_str)

#         time = now.time()
#         time_str = time.strftime('%H:%M')
#         time_dict = dict(time=time_str)
#         img_bytes = io.BytesIO(img.read())
#         img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
#         # disease= dict(context['disease_names'])
#         # print("disease dict")
#         # print(disease)
#         # Convert the integer key in the probability dictionary to a string.
#         # context["probability"] = {str(k): v for k, v in context["probability"].items()}
#         # context["disease"] = {str(k): v for k, v in context["disease"].items()}
#         # Save the data to the MongoDB database.
#         user_data = {
#             'username': username,
#             'blood_group': blood_group,
#             'work_condition': work_condition,
#             'city': city,
#             'age': age,
#             'image': img_base64,
#             'date': date_dict,
#             'time': time_dict,
#             'disease': trueorfalse,
#             'probability': proba
#             # 'disease': context["disease"],
#             # 'probability': context["probability"]
#             #all same except we dont have to do sort_convert or separate image processing as i have already done it in my code so i just need to grab true or false values perfectly
            
#         }
#         print(user_data)
#         try:
#             db.user_inputs.insert_one(user_data)
#         except Exception as e:
#             # Handle any errors that may occur.
#             print(e)
        
#         return HttpResponseRedirect('/dashboard')

import os
import cv2
import numpy as np
import pymongo
import datetime
import base64
import io
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseRedirect
from keras.models import load_model
from collections import deque

@csrf_exempt
def save_data(request):
    if request.method == 'POST':
        blood_group = request.POST['blood_group']
        work_condition = request.POST['work_condition']
        city = request.POST['city']
        age = request.POST['age']
        img = request.FILES['image']

        # Load the ML model
        ML_MODELS_DIR = settings.ML_MODELS_DIR
        model1 = load_model(os.path.join(ML_MODELS_DIR, 'model.h5'))

        # Analyze the uploaded image
        trueorfalse, proba, imagee, df = analyze_images(img, model1)
        context=sort_convert(proba)
        # To store in MongoDB
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['TempUser']

        # Get the phone number of the logged-in user
        username = request.session['username']

        # Get the current date and time.
        now = datetime.datetime.now()

        date = now.date()
        date_str = date.isoformat()
        date_dict = dict(date=date_str)

        time = now.time()
        time_str = time.strftime('%H:%M')
        time_dict = dict(time=time_str)
        img_bytes = io.BytesIO(img.read())
        
        probab=numpy_ndarray_to_dict(proba)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        img_buffer = io.BytesIO()
        imagee.save(img_buffer, format='JPEG')
        processed_image_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        # Convert DataFrame to JSON string
        df_json = df.to_json()
        # disease= dict(context['disease_names'])
        # print("disease dict")
        # print(disease)
        # Convert the integer key in the probability dictionary to a string.
        # context["probability"] = {str(k): v for k, v in context["probability"].items()}
        # context["disease"] = {str(k): v for k, v in context["disease"].items()}
        # Save the data to the MongoDB database.
        user_data = {
            'username': username,
            'blood_group': blood_group,
            'work_condition': work_condition,
            'city': city,
            'age': age,
            'image': img_base64,
            'processed_image':processed_image_base64,
            'df_data':df_json,
            'date': date_dict,
            'time': time_dict,
            'disease': trueorfalse,
            'probability': str(context['probability'])  # Ensure probability is JSON serializable
        }
        try_analyze=str(context['probability'])
        print(try_analyze)
        
        

        # try_analyze=dict(try_analyze)
        # print(try_analyze)
        try:
            db.user_inputs.insert_one(user_data)
        except Exception as e:
            # Handle any errors that may occur
            print(e)

        return HttpResponseRedirect('/dashboard')

def preprocess_image(image_file):
    # Read the image file into a numpy array
    image = np.asarray(bytearray(image_file.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    print(image)
    df, imagee=object_detection(image)
    imagee.show()
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to the required size (128x128)
    resized_image = cv2.resize(image_rgb, (128, 128))

    # Normalize the pixel values to be between 0 and 1
    normalized_image = resized_image.astype("float32") / 255.0

    # Expand dimensions to match the input shape of the model
    expanded_image = np.expand_dims(normalized_image, axis=0)

    return expanded_image, imagee, df

def analyze_images(image_file, model, threshold=1):
    Q = deque(maxlen=128)

    # Preprocess the image
    processed_image, imagee, df = preprocess_image(image_file)

    # Make predictions on the preprocessed image
    preds = model.predict(processed_image)[0]
    Q.append(preds)

    # Perform prediction averaging over the current history of previous predictions
    results = np.array(Q).mean(axis=0)
    label = results > 0.65
    probability = results

    if label:
        return True, probability, imagee, df
    else:
        return False, probability, imagee, df






#register function
def create_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['TempUser']
        hashed_password = make_password(password)

        existing_user = db.users.find_one({'username': username})
        if existing_user:
            messages.success(request,'Username already exists. Please login')
            return render(request, 'login.html')
        # Create a new user document.
        user = {
            'username': username,
            'password': hashed_password,
        }
        request.session['username'] = username
        db.users.insert_one(user)

        messages.success(request,'User created successfully!')
        return HttpResponseRedirect('/dashboard')
    return render(request, 'login.html')


# login function
def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['TempUser']
        
        try:

            user = db.users.find_one({'username': username})

            if user: 
                # Retrieve the stored hashed password from the user document
                stored_password = user['password']
                # Hash the input password using the same salt as the stored password
                if check_password(password , stored_password):
                    request.session['username'] = username
                    messages.success(request,"Correct Password") 
                    return HttpResponseRedirect('/dashboard')  # Passwords match, user is authenticated
                else:
                    messages.success(request,"Incorrect Password or User does not exist")  # Passwords do not match
                    return render(request, 'login.html')
        except Exception as e:
            messages.error(request, str(e))

    else:
        return render(request, 'login.html')


#logout Function

def logout_user(request):
    logout(request)

    if 'username' in request.session:
        del request.session['username']

    messages.success(request, "You Have Been Logged Out...")
    return HttpResponseRedirect('/')


# views.py

from bson.objectid import ObjectId
import pandas as pd
def object_detection_result_view(request, data_id):
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['TempUser']
    user_data = db.user_inputs.find_one({'_id': ObjectId(data_id)})

    if user_data:
        processed_image_base64 = user_data.get('processed_image', '')
        df_data = user_data.get('df_data', '')

        df_html = pd.read_json(df_data).to_html(classes='table table-striped table-bordered', index=False)

        return render(request, 'object_detection_result.html', {
            'processed_image_base64': processed_image_base64,
            'df_html': df_html
        })
    else:
        return HttpResponse("Data not found", status=404)
