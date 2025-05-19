import cv2
import numpy as np
import os

# Path to your XML file
helmet_cascade_path = r"C:\Users\Lenovo\Downloads\haarcascade_helmet.xml"

# Verify the file exists
if not os.path.exists(helmet_cascade_path):
    print(f"Error: File {helmet_cascade_path} not found. Check the path.")
    exit()

# Load the cascade classifier
helmet_cascade = cv2.CascadeClassifier(helmet_cascade_path)
if helmet_cascade.empty():
    print(f"Error: Failed to load {helmet_cascade_path}. Ensure it’s a valid Haar cascade XML.")
    exit()

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Trying fallback image mode...")
    # Fallback: Load a test image (replace with your image path if available)
    frame = cv2.imread(r"C:\Users\Lenovo\Downloads\test_helmet.jpg")
    if frame is None:
        print("Error: No webcam or test image available. Please provide an image path.")
        exit()
    use_webcam = False
else:
    use_webcam = True

# Main loop
while True:
    if use_webcam:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better detection
    gray = cv2.equalizeHist(gray)

    # Detect helmets
    helmets = helmet_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,    # Smaller value for more precision
        minNeighbors=3,      # Lower for more detections, increase if too noisy
        minSize=(50, 50),    # Adjust based on helmet size in frame
        maxSize=(200, 200)   # Limit false positives
    )

    # Count helmets
    helmet_count = len(helmets)

    # Draw rectangles and labels
    for (x, y, w, h) in helmets:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Helmet', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display count and warning
    cv2.putText(frame, f'Helmets: {helmet_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if helmet_count == 0:
        cv2.putText(frame, 'NO HELMET DETECTED', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Helmet Detection', frame)

    # Debug output
    print(f"Frame processed - Helmets detected: {helmet_count}")

    # Exit on 'q' (for webcam) or wait briefly (for image)
    key = cv2.waitKey(1 if use_webcam else 500) & 0xFF
    if key == ord('q'):
        break
    if not use_webcam:  # For image mode, loop the same frame
        continue

# Cleanup
if use_webcam:
    cap.release()
cv2.destroyAllWindows()