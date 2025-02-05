
## Overview

This project aims to detect whether an image is real or AI-generated using deep learning models. The detection system is based on three different approaches:

1. **CNN-based model**
2. **MobileNetV2-based model (with custom augmentation and tuning)**
3. **MobileNetV3-based model (with additional enhancements and optimizations)**

## Installation & Dependencies

To run this project, ensure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Required Libraries:

- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL (Pillow)

These dependencies should be listed in the `requirements.txt` file, which can be generated from the notebooks if not provided.

## Dataset

The dataset used for training and evaluation is from the **Vista '25 competition** on Kaggle. Users may need to download the dataset separately if they wish to retrain the models.

## Model & Training

The models are pre-trained but require preprocessing for testing. The preprocessing steps involve resizing images, normalization, and data augmentation.

### **Base Model Augmentation & Enhancements**

- **Basic CNN Model**: A simple convolutional neural network trained on preprocessed image data.
- **MobileNetV2 Model**:
  - Utilizes **MobileNetV2** as a base model.
  - Features **custom augmentations** such as random cropping, rotation, and brightness adjustments.
  - Fine-tuned on the dataset with additional dropout layers for better generalization.
- **MobileNetV3 Model**:
  - Uses **MobileNetV3** as the backbone.
  - Employs **advanced augmentation techniques**, including mixup and cutmix, to improve robustness.
  - Optimized with learning rate scheduling and additional batch normalization layers.

## Running the Project

### **Inference (Testing an Image)**

To test the model with an image, follow these steps:

1. Ensure the dataset is preprocessed correctly.
2. Run the appropriate script to load the trained model and test on an image.

Example usage:

```bash
python test_model.py --image path/to/image.jpg
```

### **Training the Model**

If you wish to train the models from scratch, run:

```bash
python train_model.py
```

Ensure that the dataset is placed correctly and the preprocessing steps are applied before training.

## Evaluation & Performance

The models were evaluated on a test dataset of **12,000 images**. The results are as follows:

| Model       | Log Loss |
| ----------- | -------- |
| Basic CNN   | 0.3358   |
| MobileNetV2 | 0.3856   |
| MobileNetV3 | 0.4990   |

## Contributors

- Aryesh
- Ashustosh Anand
- Ankush Maiti

## License

This project is open-source.&#x20;
