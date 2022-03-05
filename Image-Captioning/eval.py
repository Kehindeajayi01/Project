from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from extract_features import get_args, split_data
import cv2
import argparse
from train import load_descriptions, load_features
import ntpath


def extractFeatures(model, filename):
    
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

# generate a description for an image
def generate_desc(model, tokenizer, patentFigure, index_word, max_length, beam_size=5):

    captions = [['start', 0.0]]
    # seed the generation process
    # iterate over the whole length of the sequence
    for i in range(max_length):
        all_caps = []
        # expand each current candidate
        for cap in captions:
            sentence, score = cap
            # if final word is 'end' token, just add the current caption
            if sentence.split()[-1] == 'end':
                all_caps.append(cap)
                continue
            # integer encode input sequence
            sequence = tokenizer.texts_to_sequences([sentence])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next words
            y_pred = model.predict([patentFigure,sequence], verbose=0)[0]
            # convert probability to integer
            yhats = np.argsort(y_pred)[-beam_size:]

            for j in yhats:
                # map integer to word
                word = index_word.get(j)
                # stop if we cannot map the word
                if word is None:
                    continue
                # Add word to caption, and generate log prob
                caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
                all_caps.append(caption)

        # order all candidates by score
        ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
        captions = ordered[:beam_size]

    return captions

# evaluate the skill of the model on all the test sets
def evaluate_model(model, descriptions, patentFigures, tokenizer, index_word, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, description in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, patentFigures[key], index_word, max_length)[0]
        # store actual and predicted
        actual.append(description.split())
        # Use best caption
        predicted.append(yhat[0].split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# def evaluate_model_single_image(model, tokenizer, index_word, max_length, testImage):
#     actual, predicted = list(), list()
#     # test descriptions
#     _, test_imgs, _, test_caps = split_data()
#     test_descriptions = load_descriptions(test_imgs, test_caps)

#     # Xception model to extract features from image
#     xception_model = Xception(include_top=False, pooling="avg")
#     # extract features from test image
#     test_feats = extractFeatures(xception_model, testImage)
#     # generate description
#     captions = generate_desc(model, tokenizer, test_feats, index_word, max_length)
#     # actual test description
#     test_image_name = ntpath.basename(testImage)
#     actual_test_caption = test_descriptions[test_image_name]
#     actual.append(actual_test_caption.split())
#     print(f"===================Actual test image caption=========")
#     print(actual_test_caption)
#     print(f"============Generated Captions=======================")
#     for cap in captions:
#         # remove start and end tokens
#         seq = cap[0].split()[1:-1]
#         predicted.append(seq)
#         desc = ' '.join(seq)
#         print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))
#     print(f"========================Evaluating Generated Captions===========================") 
#     # calculate BLEU score
#     print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
#     print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
#     print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
#     print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))




if __name__ == '__main__':

    parser = get_args()
    args = parser.parse_args()
    testImage = args.testImage

    # load the tokenizer
    tokenizer = load(open('/data/kajay/Image_caption_project/tokenizer.pkl', 'rb'))
    index_word = load(open('/data/kajay/Image_caption_project/index_word.pkl', 'rb'))

    # pre-define the max sequence length (from training)
    max_length = 212

    # load model
    model = load_model("/data/kajay/Image_caption_project/Checkpoints/model-ep002-loss1.638-val_loss1.652.h5")

    # load test features (load the test features from where you saved it during training)
    test_features = load_features(features = "test")

    train_imgs, test_imgs, train_caps, test_caps = split_data()
    # test descriptions
    test_descriptions = load_descriptions(test_imgs, test_caps)

    # Xception model to extract features from image
    xception_model = Xception(include_top=False, pooling="avg")
    # extract features from test image
    test_feats = extractFeatures(xception_model, testImage)
    # generate description
    captions = generate_desc(model, tokenizer, test_feats, index_word, max_length)

    print(f"============Generated Captions=======================")
    for cap in captions:
        # remove start and end tokens
        seq = cap[0].split()[1:-1]
        desc = ' '.join(seq)
        print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))
    # print(f"===================Model Evaluation on test data====================")
    # # evaluate model
    # evaluate_model(model, test_descriptions, test_features, tokenizer, index_word, max_length)