import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define colors for different emotions
emotion_colors = {
    'happy': ((0, 255, 0), (0, 255, 0)),  # Green
    'sad': ((255, 0, 0), (255, 0, 0)),    # Blue
    'angry': ((0, 0, 255), (0, 0, 255)),  # Red
    'surprise': ((255, 255, 0), (255, 255, 0)),  # Cyan
    'neutral': ((255, 255, 255), (255, 255, 255)),  # White
    'fear': ((128, 0, 128), (128, 0, 128)),  # Purple
    #'disgust': ((0, 128, 128), (0, 128, 128)),  # Teal
}

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result['dominant_emotion'] if isinstance(result, dict) else result[0]['dominant_emotion']

            # Get the corresponding color for the emotion
            rect_color, text_color = emotion_colors.get(emotion.lower(), ((0, 255, 255), (0, 255, 255)))  # Default Yellow

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        except Exception as e:
            print(f"Error analyzing face: {e}")

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()