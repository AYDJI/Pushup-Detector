# Pushup Detector

A Python program that detects pushups using computer vision and counts them automatically.

## Features

- Real-time pushup detection using webcam
- Automatic pushup counting
- Visual feedback with angle measurements
- Body pose tracking using MediaPipe

## Requirements

- Python 3.10.0 (not tested on other versions)
- Webcam (built-in or external)

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the program:
```bash
python pushup_detector.py
```

### How to use:

1. Position yourself in front of the webcam
2. Make sure you're clearly visible and well-lit
3. Start doing pushups
4. The program will automatically detect and count your pushups
5. Press 'q' to quit the program

## How it works

The program uses MediaPipe's Pose Detection to track your body landmarks and calculates:
- Elbow angle (to detect up/down movement)
- Hip angle (to ensure proper body alignment)

Pushups are counted when:
1. Your body maintains proper alignment (hip angle > 160°)
2. Your elbows bend to the down position (angle < 90°)
3. You return to the up position (angle > 160°)

## Variables

The pushup count is stored in the `pushup_count` variable and is printed to the console in real-time. The final count is returned when the program ends.

## Troubleshooting

- If pushups aren't being detected, ensure you're positioned sideways to the camera
- Make sure there's adequate lighting
- Keep your body in a straight line during pushups
- If the camera doesn't open, check your webcam permissions

## Dependencies

- OpenCV-Python: For computer vision and video processing
- MediaPipe: For pose detection and landmark tracking
- NumPy: For mathematical calculations


