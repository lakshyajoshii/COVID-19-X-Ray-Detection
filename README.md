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

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/COVID-19-X-Ray-Detection.git](https://github.com/your-username/COVID-19-X-Ray-Detection.git)
    ```
2.  Install the required packages:
    ```bash
    pip install tensorflow keras opencv-python matplotlib seaborn pandas scikit-learn
    ```

### Usage

1.  **Training the Model:**
    The `covid_detection.py` script contains the code for training the model. You will need to have the dataset structured into `Train` and `Val` directories, each with `Covid` and `Normal` subdirectories.

2.  **Making Predictions:**
    The `project.py` script can be used to make predictions on new X-ray images. Place the images you want to test in a `Prediction` folder.

    To run a prediction:
    ```python
    python project.py
    ```
    The script will load the pre-trained model (`Detection_Covid_19.h5`) and predict the class of the sample images provided.

## Model Files

-   `best_model.keras` and `Detection_Covid_19.h5`: These are the saved, pre-trained models. You can load these files to make predictions without retraining.

## Acknowledgements

This project was developed as a tool to assist in the rapid screening of COVID-19. It is not intended to be a substitute for professional medical advice.
