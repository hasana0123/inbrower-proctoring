# import cv2
# import os
# from deepface import DeepFace

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Path to the folder containing the reference image
# data_folder = "Data\Adarsha"
# reference_image_path = os.path.join(data_folder, "IMG_8134.jpg")

# # Check if the reference image exists
# if not os.path.exists(reference_image_path):
#     print(f"Reference image not found at {reference_image_path}")
#     exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     frame = cv2.flip(frame, 1)
    
#     if not ret:
#         print("Failed to grab frame")
#         break

#     try:
#         # Perform face verification
#         result = DeepFace.verify(frame, reference_image_path, enforce_detection=False)
        
#         # Display the result on the frame
#         if result["verified"]:
#             cv2.putText(frame, "Match!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Match", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#     except Exception as e:
#         print(f"Error in face verification: {str(e)}")
#         cv2.putText(frame, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Webcam Face Matching', frame)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and destroy windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import os
from deepface import DeepFace
import time
import threading

class FaceVerifier:
    def __init__(self, reference_image_path):
        self.reference_image_path = reference_image_path
        self.is_match = None
        self.lock = threading.Lock()

    def verify_face(self, frame):
        try:
            result = DeepFace.verify(frame, self.reference_image_path, enforce_detection=False)
            with self.lock:
                self.is_match = result["verified"]
        except Exception as e:
            print(f"Error in face verification: {str(e)}")
            with self.lock:
                self.is_match = None

    def get_result(self):
        with self.lock:
            return self.is_match

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Path to the folder containing the reference image
data_folder = "Data\Adarsha"
reference_image_path = os.path.join(data_folder, "WIN_20240925_21_59_58_Pro.jpg")

# Check if the reference image exists
if not os.path.exists(reference_image_path):
    print(f"Reference image not found at {reference_image_path}")
    exit()

# Initialize FaceVerifier
face_verifier = FaceVerifier(reference_image_path)

# Variables for face verification
last_verification_time = 0
verification_interval = 1.0  # Verify every 1 second
verification_thread = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    
    if not ret:
        print("Failed to grab frame")
        break

    current_time = time.time()

    # Perform face verification at intervals
    if current_time - last_verification_time > verification_interval:
        if verification_thread is None or not verification_thread.is_alive():
            verification_thread = threading.Thread(target=face_verifier.verify_face, args=(frame.copy(),))
            verification_thread.start()
            last_verification_time = current_time

    # Get the latest result
    is_match = face_verifier.get_result()

    # Display the result on the frame
    if is_match is True:
        cv2.putText(frame, "Match!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_match is False:
        cv2.putText(frame, "No Match", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Verifying...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Face Matching', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()