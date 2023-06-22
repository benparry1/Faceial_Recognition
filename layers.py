#custom L1 distance layer module

#import dependencies
import tensorflow as tf
from tensorflow.python.keras.layers import Layer


#Siamese L1 distance class
#L1 distance layer: tells you how similar two pictures are

class L1Dist(Layer):
    
    #init mehtod - inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    #important stuff
    #tells the layer what to do when data is passed to it
    #subtracts the validation of the anchor and pos/neg embedding
    #returns an absolute value of the difference
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    

