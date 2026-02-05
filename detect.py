import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
MODEL_PATH = 'pose_landmarker_heavy.task' 

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Define Area (Central 40% of the screen)
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        color = (0, 0, 255) # Default Red
        
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                # 11 = Left Shoulder, 12 = Right Shoulder
                # Calculate the midpoint (Chest)
                chest_x = (pose[11].x + pose[12].x) / 2 * w
                chest_y = (pose[11].y + pose[12].y) / 2 * h

                # Check if Chest is inside the box
                if x1 < chest_x < x2 and y1 < chest_y < y2:
                    color = (0, 255, 0) # Green

                # Draw a "Heart" dot at the chest center
                cv2.circle(frame, (int(chest_x), int(chest_y)), 10, color, -1)

        # UI Overlay
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.imshow('Chest Detection Zone', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
