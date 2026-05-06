import cv2
import torch
from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

# Initialize the FaceDetector
face_model_path = './weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device='cuda')

# Initialize the MultitaskPredictor
multitask_model_path = './weights/MTL_backbone.pth'
multitask_model = MultitaskPredictor(model_path=multitask_model_path, device='cuda')

# Path to the input image
image_path = 'image.png'

# Detect face (returns cropped face as NumPy array and detection results)
cropped_face, dets = face_detector.get_face(image_path)

if cropped_face is not None and dets is not None:
    print("Face detected!")

    # Perform multitasking predictions
    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

    # Process emotion output
    emotion_index = torch.argmax(emotion_logits, dim=1).item()  # Get the predicted emotion index
    print(f"Predicted Emotion Index: {emotion_index}")

    # Process gaze output
    print(f"Predicted Gaze (Yaw, Pitch): {gaze_output}")

    # Process action units
    print(f"Predicted Action Units (Intensities): {au_output}")
else:
    print("No face detected.")