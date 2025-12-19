# Learn Computer Vision

A practical computer vision learning repository focused on implementation using cutting-edge deep learning frameworks. The learning materials are designed for beginners to intermediate learners who want to master concepts and applications of computer vision, particularly in object detection and facial analysis.

## 📋 Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Course Structure](#course-structure)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## ✨ Key Features

- **Deep Learning Environment Setup**: Comprehensive guide for installing TensorFlow 2.10 with GPU support
- **Real-time Emotion Detection**: Implementation of YOLOv5 for detecting emotions on human faces
- **Pre-trained Models**: Utilizes pre-trained YOLOv5 models to accelerate the learning process
- **Jupyter Notebooks**: All learning materials are available in interactive notebook format for continuous learning
- **Model Persistence**: Functions to save and load trained models for inference and fine-tuning

## 🔧 Prerequisites

Before starting, ensure your system meets the following requirements:

- **Python 3.10** (recommended for TensorFlow 2.10 with GPU)
- **pip** (Python package manager)
- **NVIDIA GPU** (optional, but recommended for optimal performance)
- **CUDA Toolkit** and **cuDNN** compatible with TensorFlow 2.10
- **Jupyter Notebook** or **JupyterLab**

### ⚠️ Compatibility Notes

This project targets **TensorFlow 2.10 with GPU**, which officially supports **Python 3.10** and requires **legacy NumPy (numpy<2)**. Newer Python or NumPy 2.x stacks are not recommended as they may cause compatibility conflicts.

### Verify Python Installation

```bash
python --version
pip --version
```

## 🚀 Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ShiroKuro017/Learn-Computer-Vision.git
cd Learn-Computer-Vision
```

### 2. Create Virtual Environment with Python 3.10

**Option A, Using Conda (Recommended):**

```bash
# Install Miniconda or Anaconda first if not already installed
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/

# Create environment with Python 3.10
conda create -n cv-tf210 python=3.10

# Activate environment
conda activate cv-tf210
```

**Option B, Using venv:**

Ensure Python 3.10 is installed on your system:

```bash
# For Windows (using py launcher)
py -3.10 -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install TensorFlow 2.10 with GPU + Legacy NumPy

Install core packages pinned for compatibility with TensorFlow 2.10:

```bash
# Install TensorFlow 2.10 and legacy NumPy
pip install "numpy<2" "tensorflow==2.10.*"

# If using NVIDIA GPU
# (verify that CUDA and cuDNN are already installed)
# For most cases, tensorflow==2.10.* includes GPU support out of the box
```

### 4. Install Computer Vision Dependencies

```bash
# Alternative deep learning framework
pip install torch torchvision torchaudio

# Object detection and image processing
pip install yolov5
pip install opencv-python

# Jupyter and data science
pip install jupyter
pip install pandas matplotlib
```

### 5. Verify Installation

Run the following snippet to verify TensorFlow GPU detection:

```python
import tensorflow as tf
import numpy as np

print("TensorFlow Version:", tf.__version__)
print("NumPy Version:", np.__version__)
print("GPU Available:", tf.test.is_built_with_cuda())
print("Physical GPUs:", len(tf.config.list_physical_devices('GPU')))

# If GPU is detected, it will display:
# GPU Available: True
# Physical GPUs: 1 (or more depending on your GPUs)
```

## 📚 Course Structure

The repository is structured in the following learning modules:

### Module 1: Environment Preparation
**File**: `Install TensorFlow w GPU.ipynb`

Learning materials include:
- Installation of TensorFlow 2.10 with GPU support
- CUDA and cuDNN configuration for TensorFlow 2.10
- Development environment verification
- Troubleshooting common installation issues
- GPU acceleration testing

### Module 2: Emotion Detection with YOLOv5
**File**: `Emotion-Detection.ipynb`

Learning materials include:
- Introduction to YOLOv5 architecture
- Real-time face detection
- Emotion feature extraction from faces
- Complete emotion detection pipeline implementation
- Functions to save and load trained models
- Performance optimization across various hardware

### Models Folder
**Directory**: `models/`

Storage for trained models that can be used for inference or further fine-tuning.

### Pre-trained Model
**File**: `yolov5s.pt`

YOLOv5 small model pre-trained on COCO dataset for object and face detection. This file is utilized in the Emotion-Detection notebook.

## 💻 How to Use

### Running Notebooks

1. **Activate Environment**:
   ```bash
   # If using Conda
   conda activate cv-tf210
   
   # If using venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Navigate to the Notebook You Want to Learn**:
   - Start with `Install TensorFlow w GPU.ipynb` for setup
   - Continue with `Emotion-Detection.ipynb` for practical emotion detection

4. **Execute Cells Sequentially**:
   - Press `Shift + Enter` to run a cell
   - Pay attention to outputs and visualizations
   - Wait for each cell to complete execution before proceeding

### Basic Usage Example

After completing the setup, you can use the model for emotion detection:

```python
# Import dependencies
import yolov5
import cv2

# Load model
model = yolov5.load('yolov5s.pt')

# Run detection on image
results = model('path/to/image.jpg')

# Display results
results.show()
```

## 🛠️ Technologies Used

| Technology | Version / Constraint | Purpose |
|-----------|----------------------|---------|
| Python | 3.10 | Primary programming language |
| TensorFlow | 2.10.x | Deep learning framework with GPU support |
| NumPy | < 2.0 | Numerical computing (legacy for compatibility) |
| PyTorch | Latest | Alternative deep learning framework |
| YOLOv5 | Latest | Object detection architecture |
| OpenCV | 4.x | Image and video processing |
| Jupyter | Latest | Interactive notebook environment |
| Pandas | Latest | Data manipulation |
| Matplotlib | Latest | Data visualization |

## 🎓 Recommended Learning Path

For optimal learning outcomes, follow this sequence:

1. **Week 1**: Environment setup and installation verification
   - Create virtual environment with Python 3.10
   - Install TensorFlow 2.10 and dependencies
   - Execute `Install TensorFlow w GPU.ipynb`
   - Ensure GPU is detected and working
   
2. **Week 2-3**: Understanding YOLOv5 concepts
   - Learn YOLOv5 architecture and methodology
   - Understand real-time object detection principles
   - Experiment with different input sizes
   
3. **Week 4-5**: Implementing Emotion Detection
   - Execute `Emotion-Detection.ipynb` thoroughly
   - Experiment with various image and video inputs
   - Learn practical model implementation details
   
4. **Week 6+**: Advanced Exploration and Development
   - Fine-tune models for specific use cases
   - Integrate into real-world applications
   - Experiment with custom datasets

## 📝 Important Notes

- **Python 3.10 Requirement**: TensorFlow 2.10 is most stable with Python 3.10. Avoid Python 3.11+ to prevent compatibility issues.
- **Legacy NumPy (numpy<2)**: NumPy 2.x may not be compatible with TensorFlow 2.10. Always use `pip install "numpy<2"`.
- **GPU vs CPU**: Training and inference processes are significantly faster with GPU. If you do not have an NVIDIA GPU, use Google Colab for free GPU access.
- **Model Size**: The `yolov5s.pt` file is several hundred megabytes. Ensure you have sufficient storage space.
- **Documentation**: Each notebook contains inline documentation and code explanations to facilitate learning.

## 🐛 Troubleshooting

### CUDA Not Available

```bash
# Verify CUDA Toolkit 11.x is installed (for TF 2.10)
nvidia-smi

# Verify TensorFlow detects GPU
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

**Solution**:
- Verify CUDA Toolkit and cuDNN installation
- Ensure NVIDIA GPU driver is updated
- For TensorFlow 2.10, use CUDA 11.2-11.8

### Import Error or Version Mismatch

```bash
# Check installed package versions
pip list | grep -i tensorflow
pip list | grep -i numpy
pip list | grep -i cuda

# Reinstall with correct constraints
pip install --force-reinstall "numpy<2" "tensorflow==2.10.*"
```

### Memory Error During Inference

- Use a smaller model (yolov5n.pt instead of yolov5l.pt)
- Reduce input image size (resize to 416x416 or smaller)
- Use smaller batch processing sizes
- Increase swap memory if RAM is limited

### NumPy 2.x Installed Automatically

If pip accidentally installs NumPy 2.x:

```bash
# Downgrade to NumPy 1.x
pip install --force-reinstall "numpy<2"

# Verify
python -c "import numpy; print(numpy.__version__)"
```

## 📂 Directory Structure

```
Learn-Computer-Vision/
├── Install TensorFlow w GPU.ipynb      # Environment setup module
├── Emotion-Detection.ipynb             # Emotion detection module
├── yolov5s.pt                          # Pre-trained YOLOv5 small model
├── models/                             # Custom model storage folder
├── README.md                           # Documentation (Indonesian)
└── README_EN.md                        # Documentation (English)
```

## 🔗 Additional References

- [TensorFlow 2.10 Documentation](https://www.tensorflow.org/versions/r2.10/api_docs)
- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- [OpenCV Tutorial](https://docs.opencv.org/)
- [Computer Vision Papers](https://arxiv.org/)
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

---

**Last Updated**: December 2025

**Environment Requirement**: Python 3.10, TensorFlow 2.10.x, NumPy<2

Happy learning! If this repository helps you, please give it a ⭐ star.
