# profake-main-project
Fake Social Media Profile Detection: A Hybrid Approach integrating Machine Learning and Deep Learning techniques  

## Motivation
The increasing prevalence of fraudulent activities and fake accounts on social media platforms inspired the creation of ProFake. These accounts not only spread misinformation but also pose risks to users’ security and trust. ProFake was developed to combat this issue by providing an efficient way to detect fraudulent Instagram accounts using advanced machine learning and deep learning techniques.

## Why Did You Build This Project?
This project was built to address the challenge of identifying fake accounts that compromise user safety and platform integrity. By integrating machine learning and deep learning methods, ProFake offers a reliable, scalable solution to this growing problem.

## Technologies used: 
Python, Google Colab, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras, Flask, and Pyngrok.  

### Data Manipulation and EDA
Pandas: Used for data manipulation, handling CSV files, and combining training and testing.  
NumPy: Utilized for numerical computations, array operations, and data transformations.  

### Visualisation
Matplotlib and Seaborn: Employed for creating visualizations such as count plots, heatmaps for correlation analysis, and confusion matrices to evaluate model performance. 

### Machine Learning and Deep Learning Frameworks
Scikit-learn: Employed for implementing traditional ML algorithms such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and MLP Classifier for classification tasks.  
TensorFlow and Keras: Utilized for building and training deep learning models like Conv1D, LSTM, Simple RNN, and Dense layers for sequential data analysis and classification.  

### Model Architecture and Fusion Techniques
Conv1D, LSTM, Simple RNN, Dense: Integrated into a sequential deep learning model architecture using Keras layers for processing sequential data and making predictions.  
Fusion Techniques: Implemented fusion strategies such as weighted averaging of predictions from traditional ML models (KNN) and DL models (LSTM) to improve overall detection accuracy.  

### Model Evaluation and Metrics
Scikit-learn metrics: Used for evaluating model performance metrics like accuracy score, confusion matrix to assess classification results, and heatmap visualization for result analysis.  

### Web Framework and Deployment
Flask: Utilized as a lightweight web framework for building the backend API to interact with the trained model and serve predictions.  
ngrok: Used for creating secure tunnels to expose the Flask web application and provide public URLs for accessing the prediction service 