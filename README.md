# 🖼️ Forgery Detection - Vision Transformer Based Image Forgery Detector

A deep learning-based image forgery detection system using Vision Transformer (ViT) combined with Error Level Analysis (ELA) to identify tampered or manipulated images.

## Features

- **Vision Transformer (ViT) Model**: Fine-tuned `google/vit-base-patch16-224` for binary classification (authentic vs. tampered)
- **Error Level Analysis (ELA)**: Detects compression inconsistencies that indicate image manipulation
- **Streamlit Web Interface**: User-friendly web app for uploading and analyzing images
- **Real-time Predictions**: Instant forgery detection with visual feedback

## Project Structure

```
forgery_detection/
├── app.py                        # Streamlit web application
├── model.py                      # ViT model architecture and ELA implementation
├── train.py                      # Training script
├── vit_forgery_finetuned.pth    # Pre-trained model weights
├── requirements.txt              # Python dependencies
├── dataset/                      # Dataset folder (not included - download separately)
│   ├── authentic/               # Authentic image samples
│   └── tampered/                # Tampered/forged image samples
└── README.md                     # This file
```

## Dataset

The image forgery dataset is stored on **Kaggle** and must be downloaded separately due to size constraints (147 MB).

**Download Dataset from Kaggle:**
- **Dataset Link:** [Image Forgery Detection Dataset - Splicing](https://www.kaggle.com/datasets/prajnar3/image-forgery-detection-dataset-splicing)
- **Size:** ~147 MB
- **Structure:** Authentic and Tampered images

After downloading, extract it to maintain the folder structure as shown above.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/forgery_detection.git
   cd forgery_detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**
   
   The dataset is hosted on Kaggle and needs to be downloaded separately due to size constraints.
   
   **Option A: Using Kaggle CLI**
   ```bash
   # Install Kaggle CLI
   pip install kaggle
   
   # Download dataset (replace DATASET_NAME with actual Kaggle dataset)
   kaggle datasets download -d [DATASET_ID]
   unzip [dataset].zip
   ```
   
   **Option B: Manual Download**
   - Visit: [Image Forgery Detection Dataset - Splicing](https://www.kaggle.com/datasets/prajnar3/image-forgery-detection-dataset-splicing)
   - Download and extract the dataset
   - Place contents in the `dataset/` folder with structure:
     ```
     dataset/
     ├── authentic/
     └── tampered/
     ```

## Usage

### Run the Web Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

**Features:**
- Upload an image (JPG, JPEG, PNG)
- View the original image
- See the ELA (Error Level Analysis) visualization
- Get a prediction: **Authentic** or **Tampered**
- View confidence score

### Train the Model (Optional)

To train on your own dataset:

```bash
python train.py
```

**Requirements:**
- Dataset structure:
  ```
  dataset/
  ├── authentic/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  └── tampered/
      ├── forged1.jpg
      ├── forged2.jpg
      └── ...
  ```

## Model Architecture

### ViTForgeryDetector

```python
- Input: RGB image (224×224)
- Backbone: Vision Transformer (ViT-Base)
  - Patch Size: 16×16
  - Hidden Dimension: 768
  - Attention Heads: 12
- Classification Head: Linear(768, 2)
- Output: [Authentic, Tampered] probabilities
```

### Error Level Analysis (ELA)

ELA detects forgeries by:
1. Re-compressing the image at a specific JPEG quality (default: 90)
2. Computing pixel-wise differences between original and re-compressed
3. Amplifying the differences for visualization
4. Feeding the ELA map to the ViT model for classification

## Dependencies

- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision utilities
- **transformers** - Hugging Face transformers library
- **streamlit** - Web app framework
- **Pillow** - Image processing
- **opencv-python** - Computer vision library
- **matplotlib** - Visualization

See `requirements.txt` for specific versions.

## Model Performance

| Metric | Value |
|--------|-------|
| Architecture | ViT-Base-Patch16-224 |
| Training Epochs | 10 |
| Batch Size | 2 |
| Optimizer | Adam (lr=1e-5) |
| Loss Function | CrossEntropyLoss |

## Technical Details

### Training Process
- **Data Loading**: Custom `ForgeryDataset` class handles authentic/tampered image loading
- **Image Preprocessing**: 
  - Resize to 224×224
  - Convert to tensor
  - Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- **Fine-tuning**: Only the classification head and feature extractor are fine-tuned on your dataset

### Inference Pipeline
1. Load image from user upload
2. Apply ELA (Error Level Analysis)
3. Preprocess ELA output to model input format
4. Forward pass through ViT model
5. Return prediction (authentic/tampered) with confidence



## Troubleshooting

### Model Not Loading
- Ensure `vit_forgery_finetuned.pth` exists in the project root
- Check that model weights match the architecture

### Low Accuracy
- Verify dataset has sufficient authentic and tampered samples
- Check data balance between classes
- Consider data augmentation in `train.py`

### Streamlit Issues
- Clear cache: `streamlit cache clear`
- Reinstall streamlit: `pip install --upgrade streamlit`


