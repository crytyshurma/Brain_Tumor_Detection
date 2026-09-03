# Brain Tumor MRI Classification

A deep learning and machine learning project for **brain tumor classification from MRI images**. The notebook experiments with multiple models and compares their classification performance on MRI brain scans.

## 📌 Project Overview

This project uses MRI images to classify brain scans into four categories:

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

The notebook explores multiple approaches, including transfer learning with **VGG16**, a custom **2D CNN**, traditional machine learning using **K-Nearest Neighbors (KNN)**, and **ResNet**.

The goal is to compare different approaches for image-based brain tumor classification and evaluate their performance using accuracy, classification reports, and confusion matrices.

## 🧠 Models Used

### 1. VGG16

A pretrained VGG16 network with ImageNet weights is used as the feature extractor.

* Input image size: `128 × 128 × 3`
* ImageNet pretrained weights
* Most VGG16 layers are frozen
* The last few layers are made trainable
* `Flatten` layer
* Dense layer with 128 neurons and ReLU activation
* Dropout for regularization
* Softmax output layer
* Adam optimizer
* Learning rate: `0.0001`
* Epochs: `5`
* Batch size: `20`

The notebook saves the trained model as `model.h5`.

### 2. 2D CNN

A separate convolutional neural network is trained for the classification task.

The notebook trains the model for five epochs and evaluates it on the testing data. The recorded test accuracy for this experiment is approximately **66.82%**.

### 3. K-Nearest Neighbors (KNN)

The notebook also experiments with a traditional machine-learning approach using KNN.

The MRI images are converted into numerical features, scaled using `StandardScaler`, and classified using `KNeighborsClassifier`.

The recorded KNN test accuracy is **92.14%**.
The trained KNN model and scaler are saved using `joblib`:

```text
knn_model.pkl
knn_scaler.pkl
```

The notebook also includes a prediction function that loads an MRI image, preprocesses it, applies the saved scaler and KNN model, and displays the predicted class with a confidence score.

### 4. ResNet

The notebook contains a separate ResNet-based experiment for MRI classification.

## 📊 Evaluation

The models are evaluated using several metrics and visualizations:

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* Training history
* Model accuracy comparison

For example, the notebook generates a confusion matrix and classification report for model predictions on the test data.

## 📈 Results

One of the model-comparison experiments records the following performance:

| Model  | Test Accuracy |
| ------ | ------------: |
| 2D CNN |    **66.82%** |
| KNN    |    **92.14%** |

The notebook also contains a final visualization comparing model accuracies in ascending order.

> **Note:** Results depend on the particular experiment, preprocessing, dataset split, and training configuration used in the notebook.

## 🗂️ Dataset Structure

The notebook expects the MRI dataset to be organized into separate training and testing directories:

```text
MRI/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The notebook loads image paths from these class directories and associates each image with its corresponding class label.

## 🛠️ Technologies Used

* Python
* Google Colab
* TensorFlow / Keras
* VGG16
* CNN
* ResNet
* Scikit-learn
* KNN
* NumPy
* Pandas
* Matplotlib
* Seaborn
* PIL
* Joblib

## 🚀 Running the Notebook

The notebook is designed to run in **Google Colab** and uses Google Drive for accessing the MRI dataset and storing trained models.

### 1. Open the notebook

Open `brain_tumor.ipynb` in Google Colab.

### 2. Enable GPU

The notebook is configured for GPU execution and specifies a **T4 GPU** in its Colab metadata.

### 3. Mount Google Drive

Run the Drive mounting cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Add the dataset

Place the MRI dataset in the expected directory structure:

```text
/content/drive/MyDrive/MRI/
```

with `Training` and `Testing` folders.

### 5. Run the notebook

Execute the cells sequentially to:

1. Load the dataset
2. Shuffle the images
3. Visualize MRI samples
4. Preprocess the images
5. Train the different models
6. Save trained models
7. Evaluate the models
8. Generate classification reports
9. Generate confusion matrices
10. Compare model accuracies

## 💾 Saved Models

The notebook saves trained models for later use.

### Deep Learning Model

```text
model.h5
```

Another 2D CNN experiment saves:

```text
2d_cnn_model.h5
```

### KNN

```text
knn_model.pkl
knn_scaler.pkl
```

The KNN model and scaler are saved using Joblib and subsequently loaded for prediction.

## 🔍 Prediction

The notebook includes functionality for predicting the tumor class of an individual MRI image.

For KNN, an input MRI image is:

```text
MRI Image
   ↓
Resize
   ↓
Convert to Grayscale
   ↓
Normalize
   ↓
Flatten
   ↓
StandardScaler
   ↓
KNN
   ↓
Predicted Class + Confidence
```

The possible output classes are:

```text
glioma
meningioma
notumor
pituitary
```

If the predicted class is `notumor`, the notebook displays **"No Tumor Detected"**; otherwise, it displays the predicted tumor class.

## 📊 Model Comparison

The notebook concludes with a model accuracy comparison visualization, allowing the different approaches to be compared based on their recorded classification accuracy.

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. The models demonstrated in this notebook should not be used as a substitute for professional medical diagnosis or clinical decision-making.

## 📁 Repository Contents

```text
.
└── brain_tumor.ipynb
```

The repository currently contains the Google Colab notebook implementing the complete experimentation workflow.
