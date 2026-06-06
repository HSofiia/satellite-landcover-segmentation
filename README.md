# Satellite Image Semantic Segmentation with U-Net

This project trains a deep learning model to automatically read satellite images and label what it sees — forests, water, roads, buildings, and background terrain — down to the level of individual pixels. Instead of a human manually drawing boundaries around each land cover type, the model learns to do this on its own after being shown thousands of labeled examples.

The practical application is land cover mapping: given a new satellite image, the model produces a color-coded map in seconds that would otherwise take hours of manual work.

## What it does

The model looks at a satellite image and assigns one of 5 categories to every single pixel:

| ID | Class | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Buildings | Red |
| 2 | Woodland | Green |
| 3 | Water | Blue |
| 4 | Road | Yellow |

The output is a color-coded mask that can be overlaid directly on the original image, making it easy to visually verify what the model detected.

## Results

The model was trained for 50 epochs and reached its best performance at epoch 38. It correctly classifies 99 out of every 100 pixels, and achieves strong overlap scores (IoU) across all 5 classes — including buildings, which are the rarest and hardest class to detect.

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 99.00% |
| Mean IoU | 0.9679 |
| Class 0 — Background IoU | 0.9852 |
| Class 1 — Buildings IoU | 0.9632 |
| Class 2 — Woodland IoU | 0.9762 |
| Class 3 — Water IoU | 0.9613 |
| Class 4 — Road IoU | 0.9534 |
| Best epoch | 38 / 50 |
| Best val loss | 0.1660 |

IoU (Intersection over Union) measures how well the predicted region overlaps with the true region — a score of 1.0 means perfect overlap.

## Project Structure

```
├── data/
│   ├── filtered_images.npy     # preprocessed image patches (2652, 256, 256, 3)
│   └── filtered_masks.npy      # corresponding segmentation masks (2652, 256, 256)
├── images/
│   └── M-33-48-A-c-4-4.tif    # source GeoTIFF satellite image
├── masks/
│   └── M-33-48-A-c-4-4.tif    # source GeoTIFF annotation mask
├── model.py                    # U-Net architecture
├── dataset.py                  # dataset class and dataloaders
├── train.py                    # training loop
├── dataset_test.ipynb          # evaluation and visualization notebook
├── unet_best.pth               # best model checkpoint (epoch 38)
├── unet_final.pth              # final model checkpoint (epoch 50)
└── train.log                   # full training log
```

## Model Architecture

The model is a U-Net — an architecture originally developed for medical image segmentation that works exceptionally well for any task where you need to label every pixel in an image.

The U-shape comes from how it processes information: it first compresses the image down step by step (encoder), forcing the network to understand what it's looking at in broad terms, then expands back up (decoder) to restore the original resolution with precise pixel-level labels. Connections between the encoder and decoder pass fine spatial detail directly across, so the model doesn't lose track of exact boundaries during compression.

```
Input (3, 256, 256)
    │
    ├── DoubleConv → 64 channels
    ├── Down → 128 channels
    ├── Down → 256 channels
    └── Down → 512 channels  (bottleneck)
    
    ├── Up + skip → 256 channels
    ├── Up + skip → 128 channels
    └── Up + skip → 64 channels
    │
Output (5, 256, 256)
```

- **7,697,605** trainable parameters
- Skip connections between encoder and decoder preserve spatial detail
- Final `Conv2d(64, 5, kernel_size=1)` outputs per-pixel class logits

## Data Preparation

The source data is a single large GeoTIFF satellite image with a corresponding hand-labeled annotation mask. GeoTIFF is a standard format for satellite and aerial imagery that stores geographic coordinates alongside pixel values.

Since neural networks work best with smaller, uniform inputs, the large image is cut into many overlapping 256×256 patches — similar to cutting a big map into tiles. The overlap (stride of 128 pixels) ensures that features near patch edges are still seen in context.

A key challenge was class imbalance: buildings cover a very small portion of the total image area, so without correction the model would simply learn to ignore them. This was addressed by keeping more copies of patches that contain rare classes in the training data.

- Patch size: **256×256**, stride: **128** (50% overlap)
- Minimum pixels threshold for rare classes: **500**
- Class 1 (buildings) patches duplicated **×5**
- Class 4 (road) patches duplicated **×3**
- Multi-class patches kept at **30%** sampling rate
- Single-class patches kept at **5%** sampling rate
- Pixel values normalized from `[0, 255]` to `[0, 1]`
- Train/val split: **80% / 20%** with `random_state=42`

Final dataset: **2,652 patches**

## Training

The model was trained on Apple M3 Pro using the MPS backend. Training for 50 epochs took approximately 40 minutes in total.

```
Epochs:        50
Batch size:    8
Optimizer:     Adam (lr=0.0003)
Loss:          CrossEntropyLoss + DiceLoss
Class weights: [0.3, 12.0, 1.0, 2.0, 4.0]
Grad clipping: max_norm=1.0
Device:        Apple MPS (M3 Pro)
```

Two loss functions are combined during training. CrossEntropy penalizes the model for each individual pixel it gets wrong. Dice Loss additionally penalizes it for predicting regions that don't overlap well with the true regions — this is especially useful for small objects like buildings that could otherwise be ignored. Class weights amplify the penalty for mistakes on rare classes, with buildings receiving the highest weight of 12.0.

Gradient clipping prevents the model's internal updates from becoming too large and destabilizing training — a problem that was observed in early runs without it.

The best model checkpoint is saved automatically whenever validation loss improves.

## Data

The satellite imagery used in this project comes from **LandCover.ai** — a publicly available dataset of high-resolution aerial images of Poland with hand-labeled land cover annotations.

- **Source:** [landcover.ai.linuxpolska.com](https://landcover.ai.linuxpolska.com)
- **Image used:** `M-33-48-A-c-4-4.tif` — RGB orthophoto
- **Mask used:** `M-33-48-A-c-4-4.tif` — corresponding hand-labeled annotation with 5 land cover classes

To run this project:

1. Download the image and mask files from the geoportal
2. Place the image in `images/` and the mask in `masks/`
3. Run the data preparation script to generate the `.npy` patch files in `data/`

The preprocessed patches (`filtered_images.npy`, `filtered_masks.npy`) are not included in this repository due to file size. Once you have the source `.tif` files, the patches can be regenerated by running the data preparation code.

## Usage

**Train the model:**
```bash
python train.py
```

**Evaluate and visualize:**

Open `dataset_test.ipynb` and run all cells. The notebook covers:
- Loading the trained model
- Visualizing predictions vs ground truth
- Computing the confusion matrix
- Pixel Accuracy and per-class IoU
- Verifying that all 5 classes are predicted

## Dependencies

```
torch
numpy
rasterio
scikit-learn
matplotlib
```

Install with:
```bash
pip install torch numpy rasterio scikit-learn matplotlib
```
