# AgroVision AI

AgroVision AI is a computer vision-based agricultural monitoring system focused on weed detection and field analysis using YOLOv8.

The project was developed to explore how AI and image processing can support precision farming by helping identify weed infestation levels from field images.

---

# Features

* Weed detection using YOLOv8
* Image quality analysis
* Infestation severity estimation
* Detection confidence analysis
* Smart recommendations based on weed density
* Streamlit-based interactive dashboard

---

# Tech Stack

* Python
* YOLOv8
* OpenCV
* Streamlit
* NumPy
* Pillow

---

# Project Structure

```bash
AgroVision-AI/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
└── sample_images/
```

---

# Setup Instructions

## 1. Clone the repository

```bash
git clone https://github.com/your-username/AgroVision-AI.git
cd AgroVision-AI
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the application

```bash
streamlit run app.py
```

---

# How It Works

1. Upload a field image
2. The system checks image quality
3. YOLOv8 performs weed detection
4. Weed count and confidence levels are calculated
5. The system estimates infestation severity
6. Recommendations are generated based on field condition

---

# Infestation Levels

| Weed Count   | Severity |
| ------------ | -------- |
| Less than 20 | Low      |
| 20 - 50      | Medium   |
| Above 50     | High     |

---

# Future Improvements

* Real-time webcam detection
* Multi-class crop and weed detection
* IoT integration for smart farming
* Mobile-friendly dashboard
* Disease detection support

---

# Note

The current model is trained primarily for weed detection and field analysis. Future versions may include crop classification and autonomous agricultural actions.

---

# Author
Devashish 
