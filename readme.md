# Skin Disease Detection and Analysis

**FIRST & LAST NAME:** Sathvika Alla

**Sparring Partner:** Farah Yakouobi


# Skin Health Detection & Analysis

## Overview
An AI-powered system for real-time detection and analysis of skin conditions and skin types using YOLOv8. The project integrates computer vision with a Raspberry Pi for interactive display and analysis, and features a Streamlit-based web interface for local use.

## 🎯 Features
- **Real-time skin condition detection** for eczema, chicken skin (keratosis pilaris), and acne
- **Skin type classification** to provide personalized analysis
- **Image and video processing** for comprehensive assessment
- **Raspberry Pi integration** for portable, interactive display
- **72% detection precision** on the test dataset
- **Streamlit web interface** for easy interaction with the system

## 🛠️ Technologies Used
- YOLOv8 for object detection and classification
- OpenCV for image processing
- Raspberry Pi for hardware integration
- Python for backend development
- Streamlit for web interface

## 📊 Performance
- Detection precision: 72%
- Real-time processing: 15-20 FPS on Raspberry Pi 4
- Supported skin conditions: Eczema, Chicken Skin, Acne

## 🖥️ Installation

```bash
# Clone the repository
git clone https://github.com/howest-mct/2023-2024-projectone-ctai-SathvikaAlla.git
cd 2023-2024-projectone-ctai-SathvikaAlla

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if not included in repository)
python download_models.py
```

## 🚀 Usage

### Streamlit Web Interface
```bash
# Launch the Streamlit application
streamlit run app.py
```
This will start a local web server and open the application in your default browser.

### Desktop Application
```bash
python main.py --source 0  # Use webcam
python main.py --source path/to/image.jpg  # Use image
python main.py --source path/to/video.mp4  # Use video
```

### Raspberry Pi Setup
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libopencv-dev python3-opencv

# Run the application
python raspberrypi_main.py
```

## 📹 Demo
Watch the live demonstration: [YouTube Demo](https://youtu.be/2CauqGtoT4o)

## 📚 Dataset
The model was trained on a custom dataset of skin conditions and types across various demographics. The dataset includes:
- Images of eczema
- Images of chicken skin (keratosis pilaris)
- Images of acne
- Balanced distribution of different skin types

## 🔄 Future Improvements
- [ ] Expand detection to more skin conditions
- [ ] Improve model accuracy with larger dataset
- [ ] Add treatment recommendations based on condition
- [ ] Develop cloud deployment options for wider accessibility
- [ ] Add mobile application support

## 📋 Project Structure
```
project/
├── models/                 # Pre-trained YOLOv8 models
├── data/                   # Dataset and annotations
├── src/                    # Source code
│   ├── detector.py         # Core detection logic
│   ├── processor.py        # Image processing utilities
│   └── classifier.py       # Skin type classification
├── raspberry_pi/           # Raspberry Pi specific code
├── utils/                  # Utility functions
├── app.py                  # Streamlit web application
├── main.py                 # Main application entry point
├── raspberrypi_main.py     # Raspberry Pi entry point
└── requirements.txt        # Project dependencies
```


