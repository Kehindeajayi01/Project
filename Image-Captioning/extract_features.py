# Import required packages
import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
from keras.applications.xception import Xception, preprocess_input
import pandas as pd
import cv2
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import json
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help = "path to the csv file")
    parser.add_argument("--imageDirectory", help = "path to the images")
    parser.add_argument("--outputDirectory", help = "path to save the extracted features")
    parser.add_argument('--testImage', help="Path to the test image")
    parser.add_argument("--train_features", help = "Path to training features")
    parser.add_argument("--test_features", help = "Path testing features")
    parser.add_argument("--checkpoint", help = "Path to save model checkpoints")
    return parser

def load_data():
    parser = get_args()
    args = parser.parse_args()
    csv_dir = args.csv
    data = pd.read_csv(csv_dir)
    return data


def split_data():
    data = load_data()
    images, captions = data["figure_file"], data["caption"]
    train_imgs, test_imgs, train_caps, test_caps = train_test_split(images, captions, random_state = 0, test_size = 0.05)
    return train_imgs, test_imgs, train_caps, test_caps


"""put the image names and description in a dictionary"""
def image_desc(images, captions):
    #data = load_data()
    #train_imgs, test_imgs, train_caps, test_caps = split_data()
    description = {}
    for img, caption in zip(images, captions):
        if img not in description:
            description[img] = caption
    
    return description

def cleanDescription(images, captions):
    description = image_desc(images, captions)
    clean_description = {}
    for img, text in description.items():
        text = text.replace(";", " ")
        text = text.replace(".", "")
        text = text.strip()
        text = text.rstrip(".;")
        clean_description[img] = text
    return clean_description

"""filter the original image files based on the image names in the dictionary"""
def filterNames(images, captions):
    common_images = []
    description = cleanDescription(images, captions)

    parser = get_args()
    args = parser.parse_args()
    image_files = os.listdir(args.imageDirectory)

    for image in image_files:
        if image in description:
            common_images.append(image)

    return common_images

"""Extract features from the images with Xception model"""
def extract_features(images, captions):
    common_images = filterNames(images, captions)
    features = {}

    parser = get_args()
    args = parser.parse_args()
    image_dir = args.imageDirectory
    try:
            model = Xception( include_top=False, pooling='avg' )
           
            for img in common_images:
                filename = os.path.join(image_dir, img)
                image = cv2.imread(filename)
                image.resize((299,299, 3))
                image = np.expand_dims(image, axis=0)
                #image = preprocess_input(image)
                image = image/127.5
                image = image - 1.0

                feature = model.predict(image)
                features[img] = feature
           
    except Exception as error:
        print(error)
    return features


 
if '__name__' == '__main__':
    parser = get_args()
    args = parser.parse_args()
    output_dir = args.outputDirectory

    # training and testing data
    train_imgs, test_imgs, train_caps, test_caps = split_data()

    # Extract features from training and testing images
    train_features = extract_features(train_imgs, train_caps)
    test_features = extract_features(test_imgs, test_caps)

    # save the training images features
    train_fp = open(os.path.join(output_dir, "train_features.p"), "wb")
    dump(train_features, train_fp)
    train_fp.close()

    # save the tesing images features
    test_fp = open(os.path.join(output_dir, "test_features.p"), "wb")
    dump(test_features, test_fp)
    test_fp.close()