from DataReader import DataSet
import glob
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, GRU,Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten, BatchNormalization, Masking
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np

word2idx, idx2word, vocab_size, max_len = DataSet()

def Image_GRU():

    embedding_size = 300
    #Image Emdedding layer
    image_model = Sequential([
    Dense(embedding_size, input_shape=(2048,), activation='relu'),
    RepeatVector(1)])

    # Caption embedding layer, get one word, output a vector
    caption_model = Sequential([
    Embedding(vocab_size, embedding_size, input_length=max_len),
    GRU(256, return_sequences=True),
    TimeDistributed(Dense(300)),
    ])
    
    final_model = Sequential([
    Merge([image_model, caption_model], mode='concat', concat_axis=1),
    Bidirectional(GRU(256, return_sequences=False)),
    Dropout(0.25),
    Dense(vocab_size),
    BatchNormalization(),
    Activation('softmax')
    ])

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    final_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print('Finish Building Model')
    print(final_model.summary())

    return image_model, caption_model, final_model

def load_weights(weight_path):
    fianl_model.load_weights(weight_path)

def predict(image, final_model):
    beam_index = 5
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = image
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

if __name__ == '__main__':
    
    # prepare data
    word2idx, idx2word, vocab_size, max_len = DataSet()
    np.save('data/idx2word', idx2word)
        
    # init GRU model
    image_GRU = Image_GRU()
    print(vocab_size, max_len)

