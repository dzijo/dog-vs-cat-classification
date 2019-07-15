from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

global model

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def predict(image):
    # load the image
    img = load_image(image)    
    # predict the class
    result = model.predict(img)
    return result[0][0].astype(int)
 
 
 
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Dog vs Cat classificator")
        self.minsize(480, 480)
 
        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
 
        self.button()
        self.buttonR()
        self.labelR = ttk.Label(self, text = "")
        self.labelR.grid(column = 0, row = 5)
 
 
 
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File", command = self.fileDialog)
        self.button.grid(column = 1, row = 1)
 
    def buttonR(self):
        self.buttonR = ttk.Button(self, text = "Predict", command = self.predict)
        self.buttonR.grid(column = 0, row = 4)
 
    def fileDialog(self): 
        self.labelR.configure(text = '')
        self.filename = filedialog.askopenfilename(initialdir =  "./", title = "Select A File", filetype =
        (("Pictures","*.jpg *.png"),("All files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)
        self.img = ImageTk.PhotoImage(Image.open(self.filename).resize((400,400)))
        self.canvas = Canvas(self.labelFrame, width = 400, height = 400)
        self.canvas.grid(column = 1, row = 3)
        self.canvas.create_image(0, 0, image = self.img, anchor = NW) 
        self.canvas.image = self.img

    def predict(self):
        self.result = predict(self.filename)
        #self.labelR = ttk.Label(self, text = "")
        #self.labelR.grid(column = 0, row = 5)
        if(self.result == 1):
            self.labelR.configure(text = 'It\'s a dog!')
        else:
            self.labelR.configure(text = 'It\'s a cat!')


# load model
model = load_model('final_model_TL.h5')
 
root = Root()
root.mainloop()