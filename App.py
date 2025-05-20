import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import LambdaCallback
import pandas as pd
import json
from streamlit_lottie import st_lottie

def load_lottie_local(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to get letter from prediction
def getLetter(result):
    classLabels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
                   10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
                   19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"

    
# Sidebar sections
st.sidebar.title("Sign Detection")
section = st.sidebar.selectbox("Choose a section", ["Home", "Data Loading & Model Training", "Predictions", "Visualizations", "Reports"])


# 👨‍💻 Developer Info in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developed By")
st.sidebar.markdown("""
- **A. Sai Theertha**  
- **D. Praneetha**  
- **K. Pranava Sai**  
- **P. Rahul**
""")

# Home section
if section == "Home":
    st.title("Sign Language Detection System")
    # Path to your Lottie JSON animation file
    lottie_file_path = "Animation - 1729326514388.json"  # Replace with your Lottie file path

    # Load and display the Lottie animation
    lottie_animation = load_lottie_local(lottie_file_path)
    st_lottie(lottie_animation, speed=1, width=700, height=400, key="home_animation")
    
    st.write("""
    Welcome to the Sign Language Detection System! This application utilizes advanced machine learning techniques 
    to recognize sign language gestures in real-time, making it a powerful tool for enhancing communication 
    between hearing and deaf communities.
    """)
    
    st.write("""
    ### Key Features:
    - **Real-time Gesture Recognition**: The application captures gestures from a live webcam feed and 
      predicts the corresponding sign language letter.
    - **User-Friendly Interface**: Designed with simplicity in mind, making it accessible for users of all 
      ages and technical backgrounds.
    - **Robust Model**: Built on a Convolutional Neural Network (CNN) that has been trained on a diverse dataset, 
      ensuring high accuracy in sign recognition.
    """)

# Data Loading & Model Training
elif section == "Data Loading & Model Training":
    st.title("Data Loading & Model Training")
    
    # Load dataset
    st.subheader("Loading Dataset")
    train = pd.read_csv("sign_mnist_train.csv")
    test = pd.read_csv("sign_mnist_test.csv")
    
    st.write("Training data preview:")
    st.write(train.head())
    
    labels = train['label'].values

    # Preprocess data
    st.subheader("Preprocessing Data")
    train.drop('label', axis=1, inplace=True)
    images = train.values
    images = np.array([np.reshape(i, (28, 28)) for i in images])
    images = np.array([i.flatten() for i in images])
    
    # Binarize labels
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=101)
    
    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    st.write("Shape of Training data:", x_train.shape)
    
    # Model Training
    st.subheader("Training Model")
    batch_size = 128
    num_classes = 24
    epochs = 10
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    
    # Create model
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Train the model with a progress bar
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size,
                        callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch+1)/epochs))])
    
    # Save the model and history to session state
    st.session_state.model = model
    st.session_state.history = history.history
    st.session_state.x_test = x_test
    st.session_state.y_test = y_test
    
    model.save('Model.keras')
    model.save('Model1.h5')
    
    st.success("Model training completed!")

# Predictions section (Live webcam)
elif section == "Predictions":
    st.title("Live Sign Prediction")
    model = load_model('Model.keras')
    run = st.checkbox('Run webcam')
    FRAME_WINDOW = st.image([])

    # Access webcam
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()

        if not ret:
            st.warning("Could not access the webcam.")
            break

        # Flip the frame horizontally for a selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB before processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks and bounding boxes if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the bounding box for the hand
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
                
                # Draw bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Extract hand region for prediction (resize it to 28x28)
                hand_region = frame[y_min:y_max, x_min:x_max]
                img_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, (28, 28))

                # Reshape and normalize the image
                img_array = img_resized.reshape(1, 28, 28, 1) / 255.0

                # Make predictions
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)
                letter = getLetter(predicted_class)

                # Overlay the prediction on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'Prediction: {letter}', (x_min, y_min - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert the frame to RGB for Streamlit and display it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    camera.release()

# Visualizations section
elif section == "Visualizations":
    st.title("Model Accuracy, Loss, and Metrics")

    # Check if history is available in session state
    if 'history' in st.session_state:
        history = st.session_state.history
        
        # Plot accuracy and loss curves
        st.subheader("Accuracy over epochs")
        
        # Plot accuracy
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['accuracy'], label='Training Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)  # Pass the figure explicitly
        
        st.subheader("Loss over epochs")
        # Plot loss
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)  # Pass the figure explicitly

        # Display confusion matrix
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test
        model = st.session_state.model
        
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred_classes)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)  # Pass the figure explicitly

        # Display accuracies
        st.subheader("Model Accuracies")
        st.write(f"Training Accuracy: {st.session_state.history['accuracy'][-1]:.2f}")
        st.write(f"Validation Accuracy: {st.session_state.history['val_accuracy'][-1]:.2f}")
        
    else:
        st.warning("Train the model first to see the visualizations.")

# Reports section
elif section == "Reports":
    st.title("Model Reports")

    # Check if model and test data are available
    if 'model' in st.session_state and 'x_test' in st.session_state and 'y_test' in st.session_state:
        model = st.session_state.model
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test

        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred_classes, target_names=[f'Class {i}' for i in range(len(set(y_true)))])
        st.text(report)
    else:
        st.warning("Train the model first to generate reports.")
