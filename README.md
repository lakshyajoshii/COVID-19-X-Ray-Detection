# COVID-19 Detection from Chest X-Ray Images

This project is a deep learning model designed to detect the presence of COVID-19 in chest X-ray images. The model is built using TensorFlow and Keras and is trained on a dataset of chest X-rays to classify them as either "Normal" or "Covid".

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

-   **Input Layer:** Accepts images of size 224x224 with 3 color channels.
-   **Convolutional Layers:** A series of `Conv2D` layers with ReLU activation to extract features from the images.
-   **Max Pooling Layers:** `MaxPooling2D` layers to downsample the feature maps.
-   **Dropout Layers:** `Dropout` layers to prevent overfitting.
-   **Flatten Layer:** Flattens the output into a one-dimensional vector.
-   **Dense Layers:** Fully connected layers for classification, with a final `sigmoid` activation function for binary output.

## Getting Started

### Prerequisites

-   Python 3.x
-   TensorFlow
-   Keras
-   OpenCV
-   Matplotlib
-   Seaborn
-   Pandas
-   Scikit-learn

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/lakshyajoshii/COVID-19-X-Ray-Detection.git](https://github.com/lakshyajoshii/COVID-19-X-Ray-Detection.git)
    cd COVID-19-X-Ray-Detection
    ```

2.  **Download the Dataset:**
    The dataset for this project is hosted on Kaggle.
    * **Download Link:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    * Click "Download" and save the `archive.zip` file.

3.  **Prepare the Dataset:**
    * Unzip the `archive.zip` file.
    * You will see a folder named `chest_xray`. Inside it are `train`, `test`, and `val` folders.
    * Create a new folder named `Dataset` in the root of this project directory.
    * Move the `train` and `val` folders from `chest_xray` into your new `Dataset` folder.
    * The final structure should look like this:
        ```
        COVID-19-X-Ray-Detection/
        ├── Dataset/
        │   ├── Train/
        │   └── Val/
        ├── covid_detection.py
        └── ... (other project files)
        ```

4.  **Install the required packages:**
    ```bash
    pip install tensorflow keras opencv-python matplotlib seaborn pandas scikit-learn
    ```

### Usage

-   **Training the Model:** The `covid_detection.py` script contains the code for training the model.
-   **Making Predictions:** The `project.py` script can be used to make predictions on new X-ray images.

## Model Files

-   `best_model.keras` and `Detection_Covid_19.h5`: These are the saved, pre-trained models. You can load these files to make predictions without retraining.

## Acknowledgements

This project was developed as a tool to assist in the rapid screening of COVID-19. It is not intended to be a substitute for professional medical advice.
