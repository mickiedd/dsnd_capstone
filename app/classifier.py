from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.layers import *
from keras.models import *
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image                  
from tqdm import tqdm
from extract_bottleneck_features import extract_InceptionV3

class DogBreedClassifier(object):
    def __init__(self):
        '''
        The init function
        Init the cascadeClassifier
        Init the InceptionV3 model and load the model
        '''
        self.InceptionV3_model = "saved_models/weights.best.InceptionV3.hdf5"
        self.cascade_model = "haarcascades/haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_model)
        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
        self.model.add(Dense(133, activation='softmax'))
        self.model.load_weights(self.InceptionV3_model)
        self.dog_names = ['01.Affenpinscher', '02.Afghan_hound', '03.Airedale_terrier', '04.Akita', 
            '05.Alaskan_malamute', '06.American_eskimo_dog', '07.American_foxhound', '08.American_staffordshire_terrier', 
            '09.American_water_spaniel', '10.Anatolian_shepherd_dog', '11.Australian_cattle_dog', '12.Australian_shepherd', 
            '13.Australian_terrier', '14.Basenji', '15.Basset_hound', '16.Beagle', '17.Bearded_collie', '18.Beauceron', 
            '19.Bedlington_terrier', '20.Belgian_malinois', '21.Belgian_sheepdog', '22.Belgian_tervuren', '23.Bernese_mountain_dog', 
            '24.Bichon_frise', '25.Black_and_tan_coonhound', '26.Black_russian_terrier', '27.Bloodhound', '28.Bluetick_coonhound', 
            '29.Border_collie', '30.Border_terrier', '31.Borzoi', '32.Boston_terrier', '33.Bouvier_des_flandres', '34.Boxer', 
            '35.Boykin_spaniel', '36.Briard', '37.Brittany', '38.Brussels_griffon', '39.Bull_terrier', '40.Bulldog', '41.Bullmastiff', 
            '42.Cairn_terrier', '43.Canaan_dog', '44.Cane_corso', '45.Cardigan_welsh_corgi', '46.Cavalier_king_charles_spaniel', 
            '47.Chesapeake_bay_retriever', '48.Chihuahua', '49.Chinese_crested', '50.Chinese_shar-pei', '51.Chow_chow', 
            '52.Clumber_spaniel', '53.Cocker_spaniel', '54.Collie', '55.Curly-coated_retriever', '56.Dachshund', '57.Dalmatian', 
            '58.Dandie_dinmont_terrier', '59.Doberman_pinscher', '60.Dogue_de_bordeaux', '61.English_cocker_spaniel', '62.English_setter', 
            '63.English_springer_spaniel', '64.English_toy_spaniel', '65.Entlebucher_mountain_dog', '66.Field_spaniel', '67.Finnish_spitz', 
            '68.Flat-coated_retriever', '69.French_bulldog', '70.German_pinscher', '71.German_shepherd_dog', '72.German_shorthaired_pointer', 
            '73.German_wirehaired_pointer', '74.Giant_schnauzer', '75.Glen_of_imaal_terrier', '76.Golden_retriever', '77.Gordon_setter', 
            '78.Great_dane', '79.Great_pyrenees', '80.Greater_swiss_mountain_dog', '81.Greyhound', '82.Havanese', '83.Ibizan_hound', 
            '84.Icelandic_sheepdog', '85.Irish_red_and_white_setter', '86.Irish_setter', '87.Irish_terrier', '88.Irish_water_spaniel', 
            '89.Irish_wolfhound', '90.Italian_greyhound', '91.Japanese_chin', '92.Keeshond', '93.Kerry_blue_terrier', '94.Komondor', 
            '95.Kuvasz', '96.Labrador_retriever', '97.Lakeland_terrier', '98.Leonberger', '99.Lhasa_apso', '00.Lowchen', '01.Maltese', 
            '02.Manchester_terrier', '03.Mastiff', '04.Miniature_schnauzer', '05.Neapolitan_mastiff', '06.Newfoundland', '07.Norfolk_terrier', 
            '08.Norwegian_buhund', '09.Norwegian_elkhound', '10.Norwegian_lundehund', '11.Norwich_terrier', '12.Nova_scotia_duck_tolling_retriever', 
            '13.Old_english_sheepdog', '14.Otterhound', '15.Papillon', '16.Parson_russell_terrier', '17.Pekingese', '18.Pembroke_welsh_corgi', 
            '19.Petit_basset_griffon_vendeen', '20.Pharaoh_hound', '21.Plott', '22.Pointer', '23.Pomeranian', '24.Poodle', '25.Portuguese_water_dog', 
            '26.Saint_bernard', '27.Silky_terrier', '28.Smooth_fox_terrier', '29.Tibetan_mastiff', '30.Welsh_springer_spaniel', 
            '31.Wirehaired_pointing_griffon', '32.Xoloitzcuintli', '33.Yorkshire_terrier']
        print(self.model)
    
    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, filename):
        '''
        INPUT:
        filename - filename path from user upload
        OUTPUT:
        bool value - check if this is a human face
        '''
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def path_to_tensor(self, img_path):
        '''
        INPUT:
        img_path - filename path from user upload
        OUTPUT:
        A 4d tensor for the classifier model
        '''
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def GuessWhatIsThis(self, filename):
        '''
        INPUT:
        filename - filename path from the user upload
        OUTPUT:
        message - A message tell the information about the image
        '''
        img = self.path_to_tensor(filename)
        feature = extract_InceptionV3(img)
        index = np.argmax(self.model.predict(feature))
        isHumanFace = self.face_detector(filename)
        message = ""
        if isHumanFace:
            message = "I think that's a human face, but at the same time, it just look like a this kind of dog: "
        else:
            message += "I think that's a "
        if index >=0 and index < len(self.dog_names):
            dog_name_split = self.dog_names[index].split(".")
            message += dog_name_split[len(dog_name_split) - 1]
        return message
        
    def predict(self, filename):
        '''
        INPUT:
        filename - filename path from the user upload
        OUTPUT:
        message - message tell the user the prediction of this image
        '''
        message = self.GuessWhatIsThis(filename)
        return message
