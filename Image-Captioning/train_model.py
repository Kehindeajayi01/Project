# import load_data as ld
from train import createTokenizer, maxLength, data_generator, CaptionModel
from extract_features import get_args, split_data
from keras.callbacks import ModelCheckpoint
from pickle import dump
import os

def train_model(weight = None, epochs = 20):
    parser = get_args()
    args = parser.parse_args()
    output_dir = args.outputDirectory
    checkpoint_path = args.checkpoint
    # load dataset
    train_imgs, test_imgs, train_caps, test_caps = split_data()
    
    # prepare tokenizer
    tokenizer = createTokenizer(train_imgs, train_caps)
    # save the tokenizer
    dump(tokenizer, open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb'))
    # index_word dict
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    # save dict
    dump(index_word, open(os.path.join(output_dir, 'index_word.pkl'), 'wb'))

    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # determine the maximum sequence length
    max_length = maxLength(train_imgs, train_caps)
    print('Description Length: %d' % max_length)

    # generate model
    model = CaptionModel(train_imgs, train_caps)

    # Check if pre-trained weights to be used
    if weight != None:
        model.load_weights(weight)

    # define checkpoint callback
    filepath = os.path.join(checkpoint_path, 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                    save_best_only=True, mode='min')

    steps = len(train_caps)
    val_steps = len(test_caps)
    # create the data generator
    train_generator = data_generator(train_imgs, train_caps, features = "train")
    val_generator = data_generator(test_imgs, test_caps, features = "test")

    # fit model
    model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps, verbose=1,
            callbacks=[checkpoint], validation_data=val_generator, validation_steps=val_steps)

    try:
        model.save(os.path.join(checkpoint_path, 'Model.h5'), overwrite=True)
        model.save_weights(os.path.join(checkpoint_path, 'weights.h5'),overwrite=True)
    except:
        print("Error in saving model.")
    print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=20)