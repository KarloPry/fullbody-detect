import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Solutions available: {dir(mp.solutions)}")
except AttributeError:
    print("Error: 'solutions' attribute still missing.")
