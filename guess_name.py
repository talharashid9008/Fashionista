# make a prediction for a new image.
# from keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from keras.models import model_from_json
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
image_dir = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\model\\input_images"
# load and prepare the image
def load_image(filename):
	# load the image
	#img = load_img(filename, grayscale=True, target_size=(28, 28))
    img = load_img(filename, grayscale=True, target_size=(224, 224))    
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

#**************************************************************************
# def load_model_with_config(model_path, config_path):
#     # Load the model configuration from the config.json file
#     with open(config_path, 'r') as f:
#         model_config = f.read()
    
#     # Create the model from the loaded configuration
#     model = model_from_json(model_config)
    
#     # Load the model weights from the specified model_path
#     model.load_weights(model_path)
    
#     return model
#**************************************************************************
def predict_random_image(imagePath,num_classes=3):
    classes = ['T-Shirt','Shoes','Shorts','Shirt','Pants','Skirt','Top','Outwear','Dress','Body','Longsleeve','Undershirt','Hat','Polo','Blouse','Hoodie']
    path = imagePath
    modelPath="C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\guess_name_model\\mobilenetv2.h5"
    #modelPath="C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\guess_name_model\\tf_model.h5"
    #config_path = "C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\guess_name_model\\config.json"
    #*********
    images_list = sorted(os.listdir(image_dir))
    pbar = tqdm(total=len(images_list))
    for image_name in images_list:
        image = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        # Resize the image
        image = image.resize((224, 224))

    #*********
        # load model
        model = load_model(modelPath)
        #model = load_model_with_config(modelPath, config_path)
        #image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        input_arr = input_arr.astype('float32') / 255.
        predictions = model.predict(input_arr, verbose=0)
        # series = pd.Series(predictions[0], index=classes)
        predicted_classes = np.argsort(predictions)
        predictions.sort()

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        # Get the name of the class with the highest probability
        predicted_class_name = classes[predicted_class_index]
        # Get the top n classes with their corresponding probabilities
        class_probabilities = [(classes[i], predictions[0][i]*100) for i in range(len(classes))]
        print('classes: ')
        print(class_probabilities)
        class_probabilities = sorted(class_probabilities, key=lambda x: x[1], reverse=False)[:num_classes]
    return class_probabilities

        # #*********************************************
        # dataset = load_dataset("fashion_mnist")
        # #image = dataset["test"]["image"][0]
        # #image = Image.open("C:\\Users\\Talha\\OneDrive\\Documents\\Resnet\\dn.jpg")
        # #*********************
        # images_list = sorted(os.listdir(image_dir))
        # pbar = tqdm(total=len(images_list))
        # class_probabilities = ()
        # for image_name in images_list:
        #     image = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
        # #*********************
        #     feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        #     model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

        #     inputs = feature_extractor(image, return_tensors="pt")

        #     with torch.no_grad():
        #         logits = model(**inputs).logits

        #     k=10
        #     topk_results = logits.topk(k, dim=1)
        #     predicted_labels = topk_results.indices.squeeze().tolist()
        #     predicted_probabilities = topk_results.values.squeeze().tolist()
        #     class_probabilities = ()
        #     for label_idx, probability in zip(predicted_labels, predicted_probabilities):
        #         label = model.config.id2label[label_idx]
        #         key = label
        #         value = probability
        #         pair = (key, value)
        #         class_probabilities += (pair,)
        #     # for i in range(1, k + 1):
        #     #     key = predicted_labels[i]
        #     #     value = predicted_probabilities[i]
        #     #     pair = (key, value)
        #     #     class_probabilities += (pair,)
        # #*********************************************
        # return class_probabilities

img_path="C:\\Users\\Talha\\Downloads\\FYP-F22-108-D-Fashionista\\test_images\\test_it.png"
predict_random_image(img_path)
top_classes = predict_random_image(img_path, num_classes=10)
for i, (class_name, probability) in enumerate(top_classes):
    print(f'Top {i+1} class: {class_name}, probability: {probability:2f}')
