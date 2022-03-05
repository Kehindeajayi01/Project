from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
# from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
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
from extract_features import get_args, filterNames, cleanDescription



# load the features and  save in a single json file
def load_features():
    parser = get_args()
    args = parser.parse_args()
    all_dir = args.outputDirectory
    features_name = args.features
    all_features = open(os.path.join(all_dir, features_name + ".p"), 'rb', encoding='utf-8')
    
    common_images = filterNames()
    #selecting only needed features
    features = {k:all_features[k] for k in common_images}

    return features

def load_descriptions():
    common_images= filterNames()
    description = cleanDescription()
    captions = {}
    for image, caption in description.items():
        if image in common_images:
            # if image not in captions:
            #     captions[image] = []
            str1 = '<start> ' + caption + ' <end>'
            desc = list(str1)
            caption[image] = "".join(desc)
    return captions

# converting dictionary to clean list of descriptions
def dict_to_list():
    descriptions = load_descriptions()
    all_desc = []
    for key in descriptions.keys():
        str1 = descriptions[key]
        desc = list(str1)
        all_desc.append("".join(desc))
    
    return all_desc


#creating tokenizer class 
#this will vectorize text corpus
#each integer will represent token in dictionary


def createTokenizer():
    desc_list = dict_to_list()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    
    return tokenizer

#calculate maximum length of descriptions
def maxLength():
    desc_list = dict_to_list()
    max_length = max(len(d.split()) for d in desc_list)
    return max_length



"""data generator, used by model.fit_generator()"""
def data_generator():
    captions = load_descriptions()
    tokenizer = createTokenizer()
    max_length = maxLength()
    features = load_features()

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


"""define the captioning model"""
def CaptionModel():
    tokenizer = createTokenizer()
    max_length = maxLength()
    vocab_size = len(tokenizer.word_index) + 1
    
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,))
    drop_1 = Dropout(0.5)(inputs1)
    dense_1 = Dense(256, activation='relu')(drop_1)  # dense_1

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,))
    embedding_1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)  # embedding_1
    drop_2 = Dropout(0.5)(embedding_1)
    lstm = LSTM(256)(drop_2)

    # Merging both models
    decoder1 = add([dense_1, lstm])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    # print(model.summary())
    # plot_model(model, to_file='/data/kajay/Image_caption_project/model.png', show_shapes=True)

    return model


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    output_dir = args.outputDirectory

    # tokenizer = createTokenizer()
    # max_length = maxLength()
    # vocab_size = len(tokenizer.word_index) + 1
    # # train our model
    # print('Vocabulary Size:', vocab_size)
    # print('Description Length: ', max_length)

    model = CaptionModel()
    train_descriptions = load_descriptions()

    epochs = 30
    steps = len(train_descriptions)
    batch_size = 32
    print("=======Training Model========")
    for i in range(epochs):
        generator = data_generator()
        model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps // 32, verbose=1)
    print("Saved Model")
    model.save(os.path.join(output_dir, "model"  + ".h5"))
       