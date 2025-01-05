
# Strong Baseline

This repository contains a Jupyter Notebook implementing watermark classification and decoding using ConvNets.

---

## Setup Instructions

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision matplotlib numpy
```

### 2. Prepare the Dataset

You can download the zip of the dataset at: https://drive.google.com/file/d/1pXP8HH4EW9l3JThleNr05d8jz96jEoIJ/view?usp=sharing

Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as `strong_baseline.py`.


### 4. Include Watermarking Model

You can download the zip of the models at: https://drive.google.com/file/d/1Jh4WotI3U0hwWDw96xgg93DRo2OS2J-F/view?usp=sharing

Unzip the pretrained watermarking model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `strong_baseline.py`
Make sure that the files: `stegastamp_wm.py`, `wm_stegastamp_decoder.pth`, and `wm_stegastamp_encoder.pth` is in the `models` directory.

### 5. Run the notebook
Open the notebook and run all the cells to see outputs.

---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Loads images
    - Loads watermarked images
2. **Model**
    - The watermarked images are generated using a Stegastamp (ConvNet) encoder.
    - A ConvNet based classifier is trained to discriminate between watermarked and not watermarked images.
    - For each image, the classifier predicts if it is watermarked or not.
    - For each watermarked image, the model uses the Stegastamp (ConvNet) decoder to decode a signature from the image.
3. **Model Evaluation**:
    - Compute metrics F1, Precision, Recall to evaluate classification of watermarked images.
    - Compute bit accuracy for watermark prediction
4. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.
