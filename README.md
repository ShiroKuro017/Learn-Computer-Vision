# Learn Computer Vision

A practical computer vision learning repository focused on implementation using cutting-edge deep learning frameworks. The learning materials are designed for beginners to intermediate learners who want to master concepts and applications of computer vision, particularly in object detection and facial analysis.

## 📋 Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Course Structure](#course-structure)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)
- [Contribution](#contribution)

## ✨ Key Features

- **Deep Learning Environment Setup**: Comprehensive guide for installing TensorFlow with GPU support
- **Real-time Emotion Detection**: Implementation of YOLOv5 for detecting emotions on human faces
- **Pre-trained Models**: Utilizes pre-trained YOLOv5 models to accelerate the learning process
- **Jupyter Notebooks**: All learning materials are available in interactive notebook format for continuous learning
- **Model Persistence**: Functions to save and load trained models for inference and fine-tuning

## 🔧 Prerequisites

Before starting, ensure your system meets the following requirements:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **NVIDIA GPU** (optional, but recommended for optimal performance)
- **CUDA Toolkit** and **cuDNN** (if using GPU)
- **Jupyter Notebook** or **JupyterLab**

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

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Open the notebook `Install TensorFlow w GPU.ipynb` and follow the installation steps to configure TensorFlow with GPU or CPU support based on your requirements.

Alternatively, install dependencies manually:

```bash
pip install tensorflow
pip install torch torchvision torchaudio
pip install yolov5
pip install jupyter
pip install opencv-python
pip install numpy pandas matplotlib
```

### 4. Verify Installation

Run the following snippet to verify TensorFlow GPU detection:

```python
import tensorflow as tf
print("GPU Available:", tf.test.is_built_with_cuda())
print("Physical GPUs:", len(tf.config.list_physical_devices('GPU')))
```

## 📚 Course Structure

The repository is structured in the following learning modules:

### Module 1: Environment Preparation
**File**: `Install TensorFlow w GPU.ipynb`

Learning materials include:
- Installation of TensorFlow with GPU support
- CUDA and cuDNN configuration
- Development environment verification
- Troubleshooting common installation issues

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

1. **Activate Virtual Environment**:
   ```bash
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

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Primary programming language |
| TensorFlow | 2.x | Deep learning framework |
| PyTorch | Latest | Alternative deep learning framework |
| YOLOv5 | Latest | Object detection architecture |
| OpenCV | 4.x | Image and video processing |
| Jupyter | Latest | Interactive notebook environment |
| NumPy | Latest | Numerical computing |
| Pandas | Latest | Data manipulation |
| Matplotlib | Latest | Data visualization |

## 🎓 Recommended Learning Path

For optimal learning outcomes, follow this sequence:

1. **Week 1**: Environment setup and installation verification
   - Execute `Install TensorFlow w GPU.ipynb`
   - Ensure all dependencies are properly installed
   
2. **Week 2-3**: Understanding YOLOv5 concepts
   - Learn YOLOv5 architecture and methodology
   - Understand real-time object detection principles
   
3. **Week 4-5**: Implementing Emotion Detection
   - Execute `Emotion-Detection.ipynb` thoroughly
   - Experiment with various image and video inputs
   
4. **Week 6+**: Advanced Exploration and Development
   - Fine-tune models for specific use cases
   - Integrate into real-world applications
   - Experiment with custom datasets

## 📝 Important Notes

- **GPU vs CPU**: Training and inference processes are significantly faster with GPU. If you do not have an NVIDIA GPU, use Google Colab for free GPU access.
- **Model Size**: The `yolov5s.pt` file is several hundred megabytes. Ensure you have sufficient storage space.
- **Documentation**: Each notebook contains inline documentation and code explanations to facilitate learning.
- **Compatibility**: Verify that your library versions are compatible with those used in the notebooks.

## 🐛 Troubleshooting

### CUDA Not Available
```bash
# Verify CUDA Toolkit installation
nvidia-smi

# Update GPU driver
# (Specific instructions depend on OS and GPU)
```

### Import Error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Memory Error During Inference
- Use a smaller model (yolov5n.pt instead of yolov5l.pt)
- Reduce input image size
- Use smaller batch processing sizes

## 📂 Directory Structure

```
Learn-Computer-Vision/
├── Install TensorFlow w GPU.ipynb      # Environment setup module
├── Emotion-Detection.ipynb             # Emotion detection module
├── yolov5s.pt                          # Pre-trained YOLOv5 model
├── models/                             # Custom model storage folder
└── README.md                           # This documentation file
```

## 🤝 Contributing

Contributions are highly welcome! Here is how to contribute:

1. **Fork** this repository
2. Create a **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### Contribution Guidelines

- Ensure code follows consistent style conventions
- Add documentation for new features
- Test thoroughly before submitting pull requests
- Write descriptive commit messages

## 🔗 Additional References

- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Tutorial](https://docs.opencv.org/)
- [Computer Vision Papers](https://arxiv.org/)

---

**Last Updated**: December 2025

Happy learning! If this repository helps you, please give it a ⭐ star.
