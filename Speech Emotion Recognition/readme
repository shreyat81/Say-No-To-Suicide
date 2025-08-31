🎙️ Speech Emotion Recognition (SER) 
- This project is a complete end-to-end Speech Emotion Recognition system that classifies    human emotions from audio recordings using machine learning. The system includes:
    1. A trained deep learning model to recognize emotions from audio
    2. A Flask-based web application with endpoints for uploading audio and receiving predictions
    3. A frontend interface to interact with the system

📂 Project Structure
├── app.py           
├── model.h5               
├── templates/
│   ├── index.html
│   ├── demo.html
│   ├── about.html
│   └── technical.html
├── uploads/                
└── README.md               

🧠 Model Details
- Input: .wav audio file (up to 4 seconds)
- Feature Extraction: 40 MFCCs with fixed length (173 frames)
- Model Type: Hybrid Model CNN+BiLSTM
- Output Classes:
    1. Angry
    2. Calm
    3. Disgust
    4. Fearful
    5. Happy
    6. Neutral
    7. Sad
    8. Surprised

🌐 Web Application Features
- Upload an audio file and get emotion prediction
- Confidence score for the predicted emotion
- Beautifully designed frontend pages:
    1. Home
    2. Technical Details
    3. Demo
    4. About

🎧 How it Works
- User uploads a .wav audio file through the web interface.
- Flask (app.py):
    1. Saves and preprocesses the audio (MFCC feature extraction)
    2. Loads the trained model and makes predictions
    3. Returns emotion and confidence
    4. Results are displayed on the frontend.

📊 Model Training
- The model was trained using MFCC features extracted from labeled datasets (e.g., RAVDESS, CREMA-D). Training includes:
- Balancing dataset 
- Data augmentation
- Normalization
- Stratified splits
- Early stopping and tuning

🎯Accuracy
Test Accuracy: 94.48%

Classification Report:
              precision    recall  f1-score   support

       angry       0.96      0.99      0.97        77
        calm       1.00      0.97      0.99        77
     disgust       0.94      0.87      0.91        77
     fearful       0.87      0.90      0.88        77
       happy       0.97      0.96      0.97        77
     neutral       0.89      0.96      0.93        77
         sad       0.93      0.96      0.94        77
   surprised       1.00      0.95      0.97        77

    accuracy                           0.94       616
   macro avg       0.95      0.94      0.94       616
weighted avg       0.95      0.94      0.94       616


