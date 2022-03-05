import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
# from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
# small library for seeing the progress of loops.
#from tqdm import tqdm_notebook as tqdm
#tqdm().pandas()
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
from extract_features import get_args, filterNames, cleanDescription, split_data



# load the features and  save in a single json file
def load_features(features = "train"):
    parser = get_args()
    args = parser.parse_args()
    all_dir = args.outputDirectory
    # training and testing data
    train_imgs, test_imgs, train_caps, test_caps = split_data()
    if features == "train":

        features_path = args.train_features
        train_features = load(open(features_path, 'rb'))
        
        common_images = filterNames(train_imgs, train_caps)
        #selecting only needed features
        train_feats = {k:train_features[k] for k in common_images}

        return train_feats 
    
    elif features == "test":
        features_path = args.test_features
        test_features = load(open(features_path, 'rb'))
        
        common_images = filterNames(test_imgs, test_caps)
        #selecting only needed features
        test_feats = {k:test_features[k] for k in common_images}

        return test_feats 


def load_descriptions(images, captions):
    common_images= filterNames(images, captions)
    description = cleanDescription(images, captions)
    captions = {}
    for image, caption in description.items():
        if image in common_images:
            #if image not in captions:
                #captions[image] = []
            str1 = '<start> ' + caption + ' <end>'
            desc = list(str1)
            captions[image] = "".join(desc)
    return captions

# converting dictionary to clean list of descriptions
def dict_to_list(images, captions):
    descriptions = load_descriptions(images, captions)
    all_desc = []
    for key in descriptions.keys():
        str1 = descriptions[key]
        desc = list(str1)
        all_desc.append("".join(desc))
    #    [all_desc.append(d) for d in descriptions[key]]
    return all_desc

#creating tokenizer class 
#this will vectorize text corpus
#each integer will represent token in dictionary


def createTokenizer(images, captions):
    desc_list = dict_to_list(images, captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    
    return tokenizer

#calculate maximum length of descriptions
def maxLength(images, captions):
    desc_list = dict_to_list(images, captions)
    max_length = max(len(d.split()) for d in desc_list)
    return max_length



"""data generator, used by model.fit_generator()"""
def data_generator(images, captions, tokenizer, max_length, features = "train"):
    captions = load_descriptions(images, captions)
    features = load_features(features = features)

    while 1:
        for key, description in captions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description, feature)
            yield ([input_image, input_sequence], output_word)

"""create input-output sequence pairs from the image description"""
def create_sequences(tokenizer, max_length, desc, feature):
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    # walk through each description for the image
    # encode the sequence
    seq = tokenizer.texts_to_sequences([desc])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        X1.append(feature)
        X2.append(in_seq)
        y.append(out_seq)
    input_img, input_seq, output_word = np.array(X1), np.array(X2), np.array(y)
    return input_img, input_seq, output_word



# def categorical_crossentropy_from_logits(y_true, y_pred):
#     y_true = y_true[:, :-1, :]  # Discard the last timestep
#     y_pred = y_pred[:, :-1, :]  # Discard the last timestep
#     loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
#                                                     logits=y_pred)
#     return loss

# def categorical_accuracy_with_variable_timestep(y_true, y_pred):
#     y_true = y_true[:, :-1, :]  # Discard the last timestep
#     y_pred = y_pred[:, :-1, :]  # Discard the last timestep

#     # Flatten the timestep dimension
#     shape = tf.shape(y_true)
#     y_true = tf.reshape(y_true, [-1, shape[-1]])
#     y_pred = tf.reshape(y_pred, [-1, shape[-1]])

#     # Discard rows that are all zeros as they represent padding words.
#     is_zero_y_true = tf.equal(y_true, 0)
#     is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
#     y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
#     y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

#     accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
#                                                 tf.argmax(y_pred, axis=1)),
#                                         dtype=tf.float32))
#     return accuracy

"""define the captioning model"""
def CaptionModel(max_length, vocab_size):
    
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    drop_1 = Dropout(0.5)(inputs1)
    dense_1 = Dense(256, activation='relu')(drop_1)  # dense_1

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    embedding_1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)  # embedding_1
    drop_2 = Dropout(0.5)(embedding_1)
    lstm = LSTM(500)(drop_2)

    # Merging both models
    decoder1 = add([dense_1, lstm])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize model
    print(model.summary())
#    plot_model(model, to_file='/data/kajay/Image_caption_project/model.png', show_shapes=True)

    return model


