# Extension 2

This repository contains a Jupyter Notebook implementing a Wformer and ViT to classify, watermark, and decoder images

---

## Setup Instructions

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision matplotlib
```

### 2. Prepare the Dataset

You can download the zip of the dataset at: https://drive.google.com/file/d/1pXP8HH4EW9l3JThleNr05d8jz96jEoIJ/view?usp=sharing

Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as `extension_1.py`.


### 4. Include Watermarking Model

You can download the zip of the models at: https://drive.google.com/file/d/1Jh4WotI3U0hwWDw96xgg93DRo2OS2J-F/view?usp=sharing

Unzip the pretrained model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `extension_2.py`

### 5. Run the notebook
Open the notebook and run all the cells to see outputs.
Note you can skip the first part and run from the section "Use Wformer and ViT Classifier to Analyze Images" if you want to skip training.

---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Loads images and captions.
2. **Model**
   - Vision Transformer Classifier
      - Defined in VisionTransformer.py, this model implements a Vision Transformer architecture tailored for image classification. This acts as an
      adversarial network to train the Wformer
   - Watermark Encoder
      - Wformer encoder and decoder is a transformer based watermarking architecture that is trained to both embed and extract watermarks from images.
3. **Training Process**:
   - Utilizes the AdamW optimizer with a learning rate of 1e-3 and adversarial learning rate of 1e-4.
   - Employs Cross Entropy Loss for classification.
   - Employs MSE loss for watermark and image loss.
4. **Model Evaluation**:
    - Compute metrics F1, Precision, Recall, and bit_acc to evaluate classification and bit accuracy of watermarked images.
5. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.