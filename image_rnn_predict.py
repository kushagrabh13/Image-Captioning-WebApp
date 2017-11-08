from image_rnn import Image_GRU,predict
from feature_extraction.FeatureExtractor_Keras import feature_extraction, feature_extraction_batch
import numpy as np
from keras.models import load_model
  
def captioning(images):
    
    # initialize GRU model
    image_model, caption_model, final_model = Image_GRU()
    #model = load_model('model/model.hdf5')
    image_model.load_weights('model/im_weights.hdf5')
    caption_model.load_weights('model/cap_weights.hdf5')
    final_model.load_weights('model/final_weights.hdf5')


    # extract feature for all images
    features = feature_extraction_batch(images)
    captions = []
    for feat in features:
        caption = predict(feat, final_model)
        captions.append(caption)
    return captions

def andro_cap(image):
    
    # initialize GRU model
    image_model, caption_model, final_model = Image_GRU()
    #model = load_model('model/model.hdf5')
    image_model.load_weights('model/im_weights.hdf5')
    caption_model.load_weights('model/cap_weights.hdf5')
    final_model.load_weights('model/final_weights.hdf5')


    # extract feature for all images
    features = feature_extraction(image)
    caption =""
    caption = predict(features, final_model)
    return caption

if __name__ == '__main__':

    img1 = 'static/img/test.jpg'
    img2 = 'static/img/dog2.jpg'
    img3 = 'static/img/dog1.jpg'

    imgs = [img1, img2, img3]
    
    captions = captioning(imgs)
    print (captions)
