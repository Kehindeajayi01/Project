from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from extract_features import get_args
from train import createTokenizer


def extract_features(model):
    parser = get_args()
    args = parser.parse_args()
    filename = args.testImage
    try:
            
        image = cv2.imread(filename)
            
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

    image.resize((299,299, 3))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer):
    tokenizer = createTokenizer()
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, patentImage, max_length):
    tokenizer = createTokenizer()
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([patentImage,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


if __name__ == '__main__':
    max_length = 200
    model_path = "/data/kajay/Image_caption_project/model.h5"
    trainedModel = load_model(model_path)
    xception_model = Xception(include_top=False, pooling="avg")
    testImageFeature = extract_features(xception_model)
    caption = generate_caption(trainedModel, testImageFeature, max_length)
    print("\n\n")
    print(caption)