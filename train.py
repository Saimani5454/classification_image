from tensorflow.keras.utils import load_img # Corrected import
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#load the image
model=VGG16()
#Load an image from file
image=load_img("bg.jpg",target_size=(224,224))
#convert the image pixels to a numpy array
image=img_to_array(image)
#reshape the data for the model
image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
#prepare the image for the VGG model
image=preprocess_input(image)
#predict the probability across all outputs classes
yhat=model.predict(image)
#convert probabilities to class labels
label=decode_predictions(yhat)
#retrieve the most likely result
label=label[0][0]
#print the classification
print('%s (%.2f%%)' % (label[1],label[2]*100))
