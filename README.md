Title: CNN-Powered Helmet Detection for Rider Safety and Law Enforcement

Description: A deep learning-based solution using Convolutional Neural Networks (CNNs) to detect whether motorcycle riders are wearing helmets. This system helps promote road safety and supports automated traffic law enforcement by analyzing images in real time or from stored data.

 📌 Features
- Real-time helmet detection using OpenCV and TensorFlow.
- Trained on labeled images of motorcyclists.

📂 Dataset
    Dataset from Kaggle:  
    [Traffic Violation Dataset V3] (🔗https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3)
    
📂 Project Structure
  Helmet_Detection_Using_CNN
├── config.py # Configuration constants
├── utils.py # Image preprocessing, logging, resizing
├── main.py # Full training pipeline + demo testing
├── test_model.py # Prediction on unseen images (test_images/)
├── test_config.py # Unit test for config file
├── test_utils.py # Unit test for utils
├── test_main.py # Unit test for main model logic
├── requirements.txt # Required Python packages
├── README.md # You're here :)
├── model/
│ └── helmet_model.h5 # Trained Keras model
├── source/
│ ├── helmet/ # Raw helmet images for augmentation
│ └── no_helmet/ # Raw no-helmet images for augmentation
├── training/
│ ├── helmet/ # Augmented helmet images
│ └── no_helmet/ # Augmented no-helmet images
├── resized/ # Uniform resized inputs for augmentation
│ ├── helmet/
│ └── no_helmet/
├── test_images/ # New images to test the trained model
│ ├── helmet1.jpg
│ └── no_helmet1.jpg
└── demo_images/
└── helmet_0.jpg # Used in main.py for testing


