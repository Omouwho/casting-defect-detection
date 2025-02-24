# Streamlit app to predict submersible pump impeller type
# !pip install streamlit
# !pip install opencv-python
# !pip install Pillow
# !pip install numpy
# !pip install tensorflow
# !pip install tensorflow-gpu
# !pip install st-annotated-text

# Importing packages
import os
import streamlit as st
import cv2
from PIL import Image 
import numpy as np 
import tensorflow as tf
from annotated_text import annotated_text

# Remove deprecation warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

model_directory = './Model/'
loaded_model_path = os.path.join(model_directory, 'model_cnn.keras')

# Function to load the image file
#@st.cache(allow_output_mutation=True)
def load_image(image_file):
    impeller_image = Image.open(image_file)
    return impeller_image

# Function to process image to the correct size   
def process_image(impeller_image):
    img = np.array(impeller_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    data = img / 255.0  # Normalize the image
    return data


# Function to load the model
def load_impellerModel():
    # Reconstructing the model architecture
    base_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    
    # Freezing the weights of the base MobileNetV2 model
    for layer in base_model.layers:
        layer.trainable = False
    
    loaded_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(units=256, activation="relu", name='dense_15'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, activation="sigmoid", name='dense_16'),
    ])
    
    # Build the model
    loaded_model.build(input_shape=(None, 256, 256, 3))
    
    # Load the model weights
    loaded_model.load_weights(loaded_model_path)
    
    return loaded_model    


# Function to predict the impeller type
def predict_impeller(reloaded_model, data):

    # predict the impeller type using the data
    image_pred = reloaded_model.predict(data)  
    
        
    # Convert values in the array from one-hot encoding to decimal numbers
    image_pred = np.squeeze(image_pred)

    # Get the impeller type from the data 
    impeller_type = "Broken" if image_pred <= 0.5 else "Good"
    return impeller_type

# Start of Program

# Setting image pixel side size
IMG_SIZE = 256

image_file = "./images/broken_good.jpeg"

st.title("Submersible Pump Impeller Type Detection")
st.markdown("""---""")

activities = ["Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == 'Detection':

    # Detection type
    detection_choice = st.radio(
        "How do you want to detect the Impeller Type?",
        ("AI Detection", "Manual Detection")
    )

    if detection_choice == "AI Detection":

        st.subheader("AI Impeller Type Detection")

        image_file = st.file_uploader("Upload your Impeller Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            # Calling the load_image function
            impeller_image = load_image(image_file)
            st.text("Original Image")			
            st.image(impeller_image, width=400)
            st.session_state["impeller_type"] = ""
            st.session_state["image_file"] = image_file

        if st.button("Process"):		
            if image_file is not None:
                # Calling the process_image function
                data = process_image(impeller_image)
                with st.spinner('Please Wait while AI Fetches the Impeller Type...'):
                    # Calling the load_model function
                    reloaded_model = load_impellerModel()

                # Calling the predict_impeller function to predict the impeller type using the data
                st.balloons()
                impeller_type = predict_impeller(reloaded_model, data)
                st.session_state["impeller_type"] = impeller_type
                
                st.markdown("""---""")
                # Display results
                if impeller_type == "Good":
                    annotated_text(
                        ("The detected impeller type is ", "#6ff"),
                        (impeller_type, "fcc"),
                    )	
                else:
                    annotated_text(
                        ("The detected impeller type is ", "#6ff"),
                        (impeller_type, "#F90060"),
                    )	
                
            else:
                st.error("Please upload a valid image of the impeller")

    if detection_choice == "Manual Detection":
        
        if "impeller_type" not in st.session_state:
            st.session_state["impeller_type"] = ""

        image_file = "./images/broken_good.jpeg"
        st.image(image_file, width=400)
        st.session_state["image_file"] = image_file

        impeller_type = st.selectbox("Select the Impeller Type", ["Broken", "Good"])		
        submit = st.button("Process")
        if submit:
            st.session_state["impeller_type"] = impeller_type
            
            # Display results
            if impeller_type == "Good":
                annotated_text(
                    ("You have Selected: ", "#6ff"),
                    (impeller_type, "fcc"),
                )	
            else:
                annotated_text(
                    ("You have Selected: ", "#6ff"),
                    (impeller_type, "#F90060"),
                )	

elif choice == 'About':	
	
	st.subheader("About Automated Defect Detection App")
	st.markdown("""---""")

	st.image(image_file, width=400)

	st.markdown("Built with Streamlit by Esther Egba")
	st.markdown("Student ID.: BI53DT")
	st.markdown("Course: MSc Data Science")
	st.markdown("School of Computer Science")	
	st.text("University of Sunderland (2023/4 - Sunderland - ASUND)")
	st.markdown("""---""")	
	st.success("Automated Defect Detection and Prediction of submersible pump impeller Webapp")


st.markdown("""---""")
st.text("Built with Streamlit and OpenCV")
