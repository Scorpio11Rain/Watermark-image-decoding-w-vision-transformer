
# Simple Baseline

This repository contains a Jupyter Notebook implementing a **naive** watermark classification and decoding baseline.
---

## Setup Instructions

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install torch torchvision transformers numpy
```

### 2. Prepare the Dataset

You can download the zip of the dataset at: https://drive.google.com/file/d/1pXP8HH4EW9l3JThleNr05d8jz96jEoIJ/view?usp=sharing

Unzip the COCO dataset.
```bash
unzip data.zip
```

Make sure that the `data` directory is in the same directory as `simple_baseline.py`.


### 4. Include Watermarking Model

Unzip the pretrained watermarking model folder.
```bash
unzip models.zip
```

Make sure that the `models` directory is in the same directory as your `simple_baseline.py`
Make sure that the files: `stegastamp_wm.py`, `wm_stegastamp_decoder.pth`, and `wm_stegastamp_encoder.pth` is in the `models` directory.

### 5. Run the notebook
Open the notebook and run all the cells to see outputs.
---

## Notebook Overview

### Key Components
1. **Data Loading**:
    - Loads images
    - Loads watermarked images for the dataset
2. **Model**
    - For each image, randomly classifies as watermarked or not watermarked.
    - For each watermarked image, randomly predicts its watermark
3. **Model Evaluation**:
    - Compute metrics F1, Precision, Recall to evaluate classification of watermarked images.
    - Compute bit accuracy for watermark prediction
4. **Hardware Optimization**:
    - Utilizes GPU (CUDA or MPS) if available for faster computations.
