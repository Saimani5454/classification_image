import streamlit as st
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
#from tensorflow.keras.utils import load_img, img_to_array # Corrected import
from tensorflow.keras.utils import img_to_array

# Load the pre-trained VGG16 model
# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_vgg16_model():
    return VGG16()

model = load_vgg16_model()

st.title("Image Classification with VGG16")

# Add a sidebar
st.sidebar.title("About the App")
st.sidebar.write("""
This app uses the pre-trained VGG16 model to classify images.
Upload an image, and the model will predict the most likely category.
""")

# Add file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Add a button to trigger prediction
predict_button = st.button("Predict")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if predict_button:
        with st.spinner("Predicting..."):
            # Convert the image to a NumPy array
            image_array = img_to_array(image)
            # Resize the image to the target size
            image_resized = Image.fromarray(image_array.astype('uint8')).resize((224, 224))
            image_array_resized = img_to_array(image_resized)
            # Reshape the data for the model
            image_reshaped = image_array_resized.reshape((1, image_array_resized.shape[0], image_array_resized.shape[1], image_array_resized.shape[2]))
            # prepare the image for the VGG model
            image_preprocessed = preprocess_input(image_reshaped)

            #predict the probability across all outputs classes
            yhat = model.predict(image_preprocessed)

            # convert probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result
            label = label[0][0]

            st.success("Prediction Complete!")
            # print the classification
            st.write("Prediction:")
            st.write('%s (%.2f%%)' % (label[1], label[2] * 100))