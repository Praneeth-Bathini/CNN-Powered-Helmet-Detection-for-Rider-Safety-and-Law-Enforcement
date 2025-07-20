Title: CNN-Powered Helmet Detection for Rider Safety and Law Enforcement

Overview: This project uses a Convolutional Neural Network (CNN) to detect whether a motorcycle rider is wearing a helmet or not. The goal is to enhance rider safety and assist in traffic law enforcement through automated image analysis.

Aim: this project aims to support smart surveillance systems in promoting road safety, reducing fatalities, and ensuring legal compliance. It serves as a real-world application of AI in intelligent traffic monitoring and rider protection.

 📌 Features
- Real-time helmet detection from images
- Data augmentation and preprocessing pipeline
- Trained CNN with high accuracy
- Demo functionality for quick testing
- Structured modular code

📂 Dataset
    Dataset from Kaggle:  
       [Traffic Violation Dataset V3] (🔗https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3)
    Classes:
       Helmet
       No Helmet
    Images are augmented using various transformations like rotation, zoom, brightness adjustment, and flips to enhance model generalization.
    
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
 └── helmet_model.h5 # Trained Keras model

├── source/
 ├── helmet/ # Raw helmet images for augmentation
 └── no_helmet/ # Raw no-helmet images for augmentation

├── training/
 ├── helmet/ # Augmented helmet images
 └── no_helmet/ # Augmented no-helmet images
 
├── resized/ # Uniform resized inputs for augmentation
 ├── helmet/
 └── no_helmet/
 
├── test_images/ # New images to test the trained model
 ├── helmet1.jpg
 └── no_helmet1.jpg
 
└── demo_images/
 └── helmet_0.jpg # Used in main.py for testing
 
🎯 What is helmet_model.h5?:
     helmet_model.h5 is the trained CNN model file that gets created when you run: python train_model.py

🧠 Model Architecture :
- CNN with 4 Convolution + MaxPooling blocks
- GlobalAveragePooling and Dense layers
- Dropout regularization
- Adam Optimizer
- Binary Crossentropy Loss
- Metrics: Accuracy, Precision, Recall, AUC

🛠️ Technologies Used :
  Languages: Python
  Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, scikit-learn, PIL
  Model: CNN-based binary classification
  Tools: VS Code, Jupyter Notebook

🚀 How to Run:
  1️⃣ Train the Model
  Triggers data augmentation → training → model save → demo on sample image : python main.py
  2️⃣ Predict on Test Images
  Place test images inside the test_images/ folder and run: python test_model.py

✅ Results: 
  Accuracy: ~95%
  Training time: ~8 minutes on CPU
  Example output:
  Prediction: 0.85 → There is likely driver without helmet!

📈 Results:
  Metric	Value:
    Accuracy	95.6%
    Precision	94.2%
    Recall	96.1%
    AUC	0.98

📚 Referred Articles & Research Papers:
This project is inspired and guided by insights from the following scholarly works and industry articles. They helped shape the architecture, model choice, and problem relevance of our helmet detection system.

1. Motorcycle Rider Helmet Detection for Riding Safety and Compliance Using CNNs (IJCRT)  
Authors: P. Kumar, A. Sharma, et al.
Published: IJCRT, 2024
Summary:
This research explores the use of Convolutional Neural Networks for detecting helmet usage in real-time. It emphasizes on using transfer learning and augmentation to boost performance on limited datasets. We adopted a similar augmentation-heavy pipeline to improve our model generalization.

2. Motorcycle Rider Helmet Detection for Riding Safety and Compliance Using CNNs (ResearchGate)
Authors: U. Nayak, K. Vora, et al.
Published: 2021
Summary:
This paper presented a practical framework for implementing helmet detection systems using CNNs. The emphasis on real-time applicability and use of simple CNN architectures helped us benchmark our initial model design and target accuracy thresholds.


🛡️ Helmet Safety Study Summary :

📌 Overview
Motorcycle accidents are a major cause of injury and death worldwide. Studies show that wearing a helmet reduces the risk of head injury by over 70% and death by 40–50%. Despite laws in many countries, compliance is inconsistent — especially in densely populated regions like India.

📊 Key Insights from Research
1. Helmet Usage Reduces Fatalities
According to WHO and national road safety studies, helmets are the single most effective way to prevent head injuries.
Non-helmeted riders are 3x more likely to suffer traumatic brain injuries.

2. Real-Time Monitoring is Essential
Manual enforcement is limited and inconsistent.
Intelligent Helmet Detection Systems using CNNs and computer vision can help automate monitoring and improve compliance.

3. Challenges Identified
Low-light conditions, occlusions, camera angles, and image quality affect detection accuracy.
Need for data augmentation and optimized CNN architectures (like MobileNet, custom CNN) for performance.

4. Policy Gaps
Despite helmet laws, enforcement is weak in many states.
Real-time surveillance systems can support smart city initiatives and road safety missions like India’s Vision Zero.

📌 Project Summary
The project presents a computer vision-based solution leveraging Convolutional Neural Networks (CNNs) to detect whether motorcycle riders are wearing helmets in real-time or from image data. It aims to enhance road safety, support traffic law enforcement, and reduce fatalities caused by non-compliance with helmet regulations.

Using a publicly available dataset of traffic images, the system is trained to classify images into two categories: Helmet and No Helmet. The images are augmented and preprocessed to ensure better generalization of the model under varying conditions such as lighting, angles, and background noise. A deep learning pipeline is constructed using TensorFlow and Keras libraries, culminating in a CNN model capable of binary classification.

The model achieves high accuracy (~95%), and can be integrated with CCTV footage, surveillance drones, or traffic monitoring systems to automatically flag violations or support analytics dashboards for city planners and police departments.
This system not only automates detection but reduces human effort in video surveillance and acts as a deterrent against traffic violations.

🔮 Future Advancements & Scope for Development
The current system works well on static images. Here’s how it can be improved and extended in real-world scenarios:

🚦 1. Real-Time Video Integration
Integrate the model with real-time video feeds from CCTV or drone footage.
Use YOLO or SSD object detection to first locate the motorcyclist and then classify the helmet status.
Raise alerts dynamically as vehicles pass through monitoring zones.

🌐 2. Vehicle & License Plate Recognition
Combine with Automatic Number Plate Recognition (ANPR) systems to track and log violating vehicle registrations.
Link to government databases for issuing automated fines or warnings.

📱 3. Mobile App for Field Officers
Develop a lightweight version of the model using TensorFlow Lite for smartphones.
Enable on-the-spot detection by traffic officers through phone cameras.

🛑 4. Violation Logging and Analytics
Build an integrated dashboard to visualize violation statistics, high-risk zones, and time-based trends.
Use this data for urban planning, policing strategies, and safety campaigns.

🔍 5. Multi-Class Detection
Extend beyond binary classification:
  Proper Helmet
  Improper Helmet (e.g., chinstrap not fastened)
  No Helmet
Train on diverse helmet types and rider postures to improve generalizability.

🤖 6. Edge Computing Deployment
Deploy on Raspberry Pi or Jetson Nano at the edge (near camera source) to reduce latency and network overhead.
Useful for remote or low-bandwidth regions.

🔐 7. Privacy-Preserving Techniques
Integrate blurred face detection to ensure personal privacy is maintained in compliance with data protection laws.
Store only relevant metadata and violation frames instead of full video feeds.

🛡️ Conclusion
This helmet detection system is a practical, scalable, and socially impactful project that demonstrates how AI can solve real-world safety issues. With further enhancements and support from smart city infrastructure, it can play a vital role in saving lives, enforcing laws, and creating safer roads for all.

🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork and submit a pull request.
