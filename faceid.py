from kivy.config import Config
Config.set('kivy','keyboard_mode','systemanddock')

#import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#build app and layout

class CamApp(App):

    def build(self):
        #main layout compnents
        self.web_cam = Image(size_hint=(1,.8))
        self.button =  Button(text="Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        #add items to layout
        #layout is in sequential order
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.button)

        #load keras model
        #self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
        
        self.model= tf.keras.models.load_model("siamesemodel")#, custom_objects = {'L1Dist' : L1Dist})

        #setup video capture device
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        #read frame from opencv
        ret, frame = self.capture.read()
       
        # Define ROI Box Dimensions
 
        top_left_x = int (540)
        top_left_y = int (300)
        bottom_right_x = int (540+250)
        bottom_right_y = int (300+250)

        # Draw rectangular window for our region of interest   
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)


        # flip horizontal and convert image to texture
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

        #preprocessing - scale and resize
    def preprocess(self, file_path):
    
        #read in image from file path
        byte_img = tf.io.read_file(file_path)
        #load in the image
        img = tf.io.decode_jpeg(byte_img)
        
        #preprocessing steps- resize the image to be 100 pixels x 100 pixels x 3 colors
        img = tf.image.resize(img, (100,100))
        #scale image to be between 0 and 1
        img = img / 255.0
    
        #return image
        return img
    
    #verification function

    def verify(self, *args):
        #specify thresholds
        detection_threshold = 0.75
        verification_threshold = 0.75
        
        #capture input image from webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[300:300+250,540:540+250,:]
        cv2.imwrite(SAVE_PATH, frame)

        #build results array
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):   #loops throught the images in verification images folder
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))   #scale and resize input image from webcam
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))  #scale and resize images in verification folder
            
            #make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        #detection threshold
        detection = np.sum(np.array(results) > detection_threshold) 
        
        #verification threshold
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        #set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'


        # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        
        return results, verified
        

if __name__ == '__main__':
    CamApp().run()