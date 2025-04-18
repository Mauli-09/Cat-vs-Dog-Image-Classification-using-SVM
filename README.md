# 🐶🐱 Cat vs Dog Image Classification using SVM

This project is a beginner-friendly machine learning pipeline that classifies images of cats and dogs using a Support Vector Machine (SVM). It demonstrates how traditional ML techniques can be applied to image data through preprocessing and feature engineering, without relying on deep learning.

---

## 🎯 Project Objective

Build a binary image classifier that can distinguish between cats and dogs using the **Kaggle Dogs vs Cats dataset** and a **Support Vector Machine (SVM)** model.

---

## 🧠 Workflow

1. 📥 **Dataset Loading**  
   Load labeled images of cats and dogs from the dataset directory.

2. 🧼 **Preprocessing**  
   - Resize all images to a fixed size (e.g., 64x64)  
   - Convert to grayscale (optional)  
   - Flatten into feature vectors  

3. 🧪 **Model Training**  
   - Use scikit-learn’s `SVC` to train a binary classifier  
   - Perform train/test split for evaluation  

4. 📊 **Model Evaluation**  
   - Calculate accuracy, precision, recall, F1-score  
   - Generate confusion matrix

---

## 📁 Dataset Details

- **Source**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- **Total Images**: 25,000 JPEGs (approx. 12,500 per class)
- **File Format**: `cat.0.jpg`, `dog.1.jpg`, etc.
- **Labels**: Extracted from file names

---

## 🛠️ Installation

Make sure you have Python 3 installed. Then install required libraries:

```bash  
pip install numpy opencv-python scikit-learn matplotlib
 How to Run the Project
Clone this repository:

bash
Copy code
git clone https://github.com/yourusername/cat-vs-dog-svm.git
cd cat-vs-dog-svm
Download the dataset from Kaggle and place it in:

bash
Copy code
datasets/train/
Open the notebook or script and run:

bash
Copy code
python cat_dog_svm.py
📈 Sample Output
text
Copy code
Accuracy: 91.2%

Confusion Matrix:
[[1140   60]
 [ 110 1190]]

Classification Report:
              precision    recall  f1-score   support

         Cat       0.91      0.95      0.93      1200
         Dog       0.95      0.91      0.93      1300

    accuracy                           0.93      2500

🧰 Tools & Libraries
Python 3

NumPy

OpenCV

Scikit-learn

Matplotlib

🔮 Future Improvements
🧪 Use Histogram of Oriented Gradients (HOG) for better features

🤖 Compare with CNN-based deep learning models

🌐 Add a Streamlit or Flask web interface

☁️ Deploy model on cloud (Heroku, Hugging Face Spaces)

👨‍💻 Author
Made with ❤️ by [Mauli]
📬 Reach me at [maulikangude007@gmail.com]
