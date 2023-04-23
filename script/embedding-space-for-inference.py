from architecture_embedding import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

######pathsandvairables#########
face_data = 'mtcnn-faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "fine_tune_model/embedding_model.h5"
face_encoder.load_weights(path)
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################

"""
"Create a dictionary of encodings/embeddings for each face in the dataset.
Sum the encodings of all the images of a person and 
normalize the sum to have a length of 1.
"""

for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        face = img_RGB.astype('float32')
            
        face_d = np.expand_dims(face, axis=0)
        face_d = preprocess_input(face_d)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode


if not os.path.exists('encodings'):
    os.mkdir('encodings')

print(encoding_dict) 


path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)






