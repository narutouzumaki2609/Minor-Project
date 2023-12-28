import json
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import collections
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Model



# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open ("dumped/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open ("dumped/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)



print("Loading the model...")
model = load_model('savedModels/model_14.h5')

resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)



# Generate Captions for a random image
# Using Greedy Search Algorithm

def predict_caption(photo):

    inp_text = "startseq"

    #max_len = 80 which is maximum length of caption
    for i in range(80):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


from io import BytesIO
from PIL import Image

# def preprocess_image(file_content):
#     # Create a BytesIO object from the file content
#     img_stream = BytesIO(file_content)

#     # Use PIL to open the image from the BytesIO object
#     img = Image.open(img_stream)

#     # Resize the image to the desired input shape
#     img = img.resize((224, 224))

#     # Convert the image to a NumPy array
#     img_array = np.array(img)

    # Expand dimensions to match the expected shape for ResNet50
    # img_array = np.expand_dims(img_array, axis=0)

    # # Normalize image according to ResNet50 requirements
    # img_array = preprocess_input(img_array)

    # return img_array

# def preprocess_image(img):
#     with open(img, 'rb') as file:
#         img = image.load_img(file, target_size=(224, 224))
#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = preprocess_input(img)
#     return img


def preprocess_image (img):
    
    img=image.load_img(img,target_size=(224, 224))
    # img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img
# import numpy as np
# from PIL import Image
# from django.core.files.uploadedfile import InMemoryUploadedFile

# def preprocess_image(img):
#     if isinstance(img, InMemoryUploadedFile):
#         # Use Django's InMemoryUploadedFile.read() method to get the content as bytes
#         img_content = img.read()
#     else:
#         # If img is already a file path, read it normally
#         with open(img, 'rb') as f:
#             img_content = f.read()

#     # Use BytesIO to handle binary data
#     img_stream = BytesIO(img_content)

#     try:
#         # Use PIL to open the image from the BytesIO object
#         img = Image.open(img_stream)
#         # Resize the image to the desired input shape
#         img = img.resize((224, 224))
#         # Convert the image to a NumPy array
#         img_array = np.array(img)
#         # Expand dimensions to match the expected shape for ResNet50
#         img_array = np.expand_dims(img_array, axis=0)
#         # Normalize image according to ResNet50 requirements
#         img_array = preprocess_input(img_array)

#         return img_array

#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return None



# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image (img):
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    # feature_vector = feature_vector.reshape((-1,))
    return feature_vector


def runModel(img_name):
    #img_name = input("enter the image name to generate:\t")

    print("Encoding the image ...")
    photo = encode_image(img_name).reshape((1, 2048))



    print("Running model to generate the caption...")
    caption = predict_caption(photo)

    img_data = plt.imread(img_name)
    plt.imshow(img_data)
    plt.axis("off")

    #plt.show()
    print(caption)
    return caption

# runModel('uploads\SAURABH.jpg')