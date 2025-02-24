# Casting Defect Detection
This project focuses on the classification of casting defects in submersible pump impellers using machine learning techniques. The program classifies the impellers into two categories: 'Defective' and 'Non-Defective'. The dataset consists of 7,348 grayscale images of size 300x300 pixels.


## Streamlit App
The Streamlit app provides an interactive interface to classify the impeller images.


### Usage
1. Navigate to the directory containing the `app.py` file.
2. Open in command prompt and run the following command `streamlit run app.py` to start the Streamlit app.
3. if you have an annotated text error, run the following command  `pip install st-annotated-text` to install annotated text.
4. Upload an image of the submersible pump impeller.
5. Click the "Process" button to get the classification result.


## Dataset
The dataset can be found on Kaggle at the following link:
- [Real-Life Industrial Dataset of Casting Product]
(https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)


## Project Structure
The project is divided into the following main components:
1. Jupyter Notebook: The notebook contains the following sections:
    - Data Preparation
    - Data Augmentation
    - Data Visualization
    - Model Building and Training
    - Model Evaluation
    - Prediction on Single Image

2. Streamlit App: The app provides a user-friendly interface for users to upload an image and get predictions on the impeller's condition.

## Jupyter Notebook
The Jupyter notebook includes:

### Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
```

### Loading and Preprocessing Data
```python
# Paths to the directories containing the images of submersible pump impeller
OK_DIR = "Dataset/casting/ok_front/"
DEF_DIR = "Dataset/casting/def_front/"

# okay is a list that contains the filenames of the 'OK' or non-defective images in the OK_DIR
okay = os.listdir(OK_DIR)

# defective is a list that contains the filenames of the 'Defective' images in the DEF_DIR
defective = os.listdir(DEF_DIR)

# Constructs the full path of each 'OK' image
okay_paths = [OK_DIR + fname for fname in okay]

# Constructs the full path of each 'Defective' image
defective_paths = [DEF_DIR + fname for fname in defective]

# Creates labels for the images (1 for 'OK' and 0 for 'Defective')
okay_labels = np.ones(len(okay_paths))  
defective_labels = np.zeros(len(defective_paths))  

# Concatenates the image paths and labels
train_paths = np.concatenate([okay_paths, defective_paths])
train_labels = np.concatenate([okay_labels, defective_labels])
```

### Data Augmentation and Loading into TensorFlow Dataset
```python
# Defines a function that takes an image path and label then returns the preprocessed image and its corresponding label
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=(256, 256))
    image = tf.cast(image, dtype=tf.float32)/255
    return image, label

# Creates a TensorFlow dataset from the training image paths and labels
train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

# Applies the load_image function to each element in the dataset to load and preprocess the images
train_dataset = train_dataset.map(load_image)

# Shuffles the training dataset
train_dataset = train_dataset.shuffle(buffer_size=500)

# Batches the dataset into batches of size 32
train_dataset = train_dataset.batch(32)

# Prefetches the data to optimize GPU memory usage
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
```

### Model Building and Training
```python
# Defines the convolutional neural network (CNN) model architecture
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiles the model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trains the model
history = model_cnn.fit(train_dataset, epochs=10)
```

### Model Evaluation and Prediction on Single Image
```python
# Loads and preprocesses a single test image
image_path = 'test_image.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img, axis=0)
img = img / 255.0  # Normalize the image

# Makes predictions on the test data
prediction = model_cnn.predict(img)

# Converts the predicted probabilities to binary class labels
if prediction > 0.5:
    prediction_label = "OK"
else:
    prediction_label = "Defective"
```

### Code Snippet

```python
# Streamlit app to predict submersible pump impeller type
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
    impeller_type = "Defective" if image_pred <= 0.5 else "OK"
    return impeller_type

# Start of Program

# Setting image pixel side size
IMG_SIZE = 256

st.title("Casting Defect Detection")
st.markdown("""---""")

activities = ["Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

# Detection type
detection_choice = st.radio(
    "How do you want to detect the Casting Defect?",
    ("AI Detection", "Manual Detection")
)

if detection_choice == "AI Detection":

    st.subheader("AI Casting Defect Detection")

    image_file = st.file_uploader("Upload your Casting Image", type=['jpg', 'png', 'jpeg'])

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
            with st.spinner('Please Wait while AI Fetches the Casting Defect Type...'):
                # Calling the load_model function
                reloaded_model = load_impellerModel()

            # Calling the predict_impeller function to predict the impeller type using the data
            st.balloons()
            impeller_type = predict_impeller(reloaded_model, data)
            st.session_state["impeller_type"] = impeller_type
            
            st.markdown("""---""")
            # Display results
            if impeller_type == "OK":
                annotated_text(
                    ("The detected casting defect type is ", "#6ff"),
                    (impeller_type, "fcc"),
                )	
            else:
                annotated_text(
                    ("The detected casting defect type is ", "#6ff"),
                    (impeller_type, "#F90060"),
                )	
             
        else:
            st.error("Please upload a valid image of the casting defect")

if detection_choice == "Manual Detection":
    
    if "impeller_type" not in st.session_state:
        st.session_state["impeller_type"] = ""

    image_file = "./images/broken_good.jpeg"
    st.image(image_file, width=400)
    st.session_state["image_file"] = image_file

    impeller_type = st.selectbox("Select the Casting Defect Type", ["Defective", "OK"])		
    submit = st.button("Process")
    if submit:
        st.session_state["impeller_type"] = impeller_type
        
        # Display results
        if impeller_type == "OK":
            annotated_text(
                ("You have Selected: ", "#6ff"),
                (impeller_type, "fcc"),
            )	
        else:
            annotated_text(
                ("You have Selected: ", "#6ff"),
                (impeller_type, "#F90060"),
            )	
```

## Installation and Setup

1. Clone the repository:
    ```
    git clone https://github.com/your-username/casting-defect-detection.git
    ```
2. Navigate to the project directory:
    ```
    cd casting-defect-detection
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebook
1. Open the Jupyter notebook `Casting_Defect_Detection.ipynb`.
2. Follow the instructions in the notebook to preprocess the data, build, and train the model.
3. Use the trained model to make predictions on new images.

### Streamlit App
1. Navigate to the directory containing the `app.py` file.
2. Run the following command to start the Streamlit app:
    ```
    streamlit run app.py
    ```
3. Upload an image of the submersible pump impeller.
4. Click the "Process" button to get the classification result.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Credits
This Project was Designed by Esther Egba - BI53DT. It is a Domain Specific Data Science Development project for Assignment 2 (CETM46) for 2023/2024 session.



