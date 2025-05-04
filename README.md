# vgg19-hybrid_deep_deepfake_deteection_model

---

## 🧠 Methodology

### 1. **Data Preprocessing**
- Extract frames from videos using `OpenCV`.
- Detect and crop faces with **MTCNN** or **RetinaFace**.
- Resize all face crops to **224×224**.

### 2. **Style-Based Path**
- Pass face crops through **VGG-19**.
- Extract feature maps from `conv1_1` to `conv5_1`.
- Compute **Gram Matrix** to capture spatial style and texture.

### 3. **Content-Based Path**
- Extract **Optical Flow** using Farneback or RAFT.
- Extract **Face Embeddings** using FaceNet or ArcFace.

### 4. **Feature Fusion**
- Concatenate style and content features.
- Apply attention layers (self-attention/spatial attention).
- Pass through a **Multi-Layer Perceptron** for classification.

---

## 📊 Model Architecture Highlights

- 🔷 **VGG-19** pretrained on ImageNet for style extraction.
- 🔷 **Gram Matrix** for style correlation detection.
- 🔷 **Optical Flow / Embeddings** for temporal/content integrity.
- 🔷 **Attention Layer** to prioritize key manipulated regions.
- 🔷 **MLP Classifier** with Dropout and BatchNorm.

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision / Recall**
- **AUC-ROC**
- **Grad-CAM Visualization** (for interpretability)

---

## 🛠 Requirements

- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- MTCNN / RetinaFace
- Scikit-learn
- Matplotlib

```bash
pip install -r requirements.txt
