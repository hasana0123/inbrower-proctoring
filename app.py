import cv2 as cv
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pyaudio
import threading
import time

app = FastAPI()

# For templates (HTML rendering)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Fake in-memory user storage
fake_user_db = {"user": {"username": "user", "password": "password"}}
current_user = None

# Face tracking setup
mp_face_mesh = mp.solutions.face_mesh

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

def eucidiean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right = eucidiean_distance(iris_center, right_point)
    total_distance = eucidiean_distance(right_point, left_point)
    gaze_ratio = center_to_right / total_distance

    if gaze_ratio < 0.4:
        return "RIGHT", gaze_ratio
    elif 0.4 <= gaze_ratio <= 0.55:
        return "CENTER", gaze_ratio
    else:
        return "LEFT", gaze_ratio
    
def detect_sound():
    global sound_detected
    # PyAudio setup for sound detection
    FORMAT = pyaudio.paInt16  # Audio format (16-bit resolution)
    CHANNELS = 1              # Mono channel
    RATE = 44100              # Sampling rate (44.1kHz)
    CHUNK = 1024              # Number of samples per chunk
    THRESHOLD = 500           # Adjust this value based on your environment

    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for sound...")

    try:
        while True:
            # Read audio data from the stream
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # Compute the average volume (amplitude)
            volume = np.linalg.norm(data)

            # Check if the volume exceeds the threshold
            if volume > THRESHOLD:
                sound_detected = True
            else:
                sound_detected = False

            time.sleep(0.1)  # Adjust frequency of sound detection

    except KeyboardInterrupt:
        print("Audio detection stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio object
    p.terminate()

# Start the audio detection in a background thread
audio_thread = threading.Thread(target=detect_sound)
audio_thread.daemon = True  # Ensure it closes with the main program
audio_thread.start()

def generate_video_feed():
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            if sound_detected:
                cv.circle(frame, (600, 50), 20, (0, 0, 255), -1)  # Red dot

            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                iris_center_left = np.array([l_cx, l_cy], dtype=np.int32)
                iris_center_right = np.array([r_cx, r_cy], dtype=np.int32)

                cv.circle(frame, iris_center_left, int(l_radius), (0, 255, 0), 1)
                cv.circle(frame, iris_center_right, int(r_radius), (0, 255, 0), 1)

                iris_pos, gaze_ratio = iris_position(iris_center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
                cv.putText(frame, iris_pos, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                
                for face_landmarks in results.multi_face_landmarks:
                    face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                rmat, _ = cv.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -7:
                    text = "Looking Down"
                elif x > 13:
                    text = "Looking Up"
                else:
                    text = "Forward"

                nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                # cv.line(frame, p1, p2, (255, 0, 0), 3)
                cv.putText(frame, text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == fake_user_db["user"]["username"] and password == fake_user_db["user"]["password"]:
        global current_user
        current_user = username
        return RedirectResponse(url="/dashboard", status_code=302)
    return {"message": "Invalid credentials"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if current_user:
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})
    return RedirectResponse(url="/")

@app.get("/video_feed")
async def video_feed():
    if current_user:
        return StreamingResponse(generate_video_feed(), media_type="multipart/x-mixed-replace; boundary=frame")
    return RedirectResponse(url="/")

# ============================ WebSocket for Audio Detection ==================================

# A list to keep track of WebSocket connections
# active_connections = []

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     active_connections.append(websocket)
#     try:
#         while True:
#             await websocket.receive_text()  # Keep the connection alive
#     except:
#         active_connections.remove(websocket)

# def notify_clients(message: str):
#     for connection in active_connections:
#         try:
#             # Send the message to each connected client
#             connection.send_text(message)
#         except:
#             active_connections.remove(connection)

# # # ============================ Audio Detection ==================================
# def detect_sound():
#     # PyAudio setup for sound detection
#     FORMAT = pyaudio.paInt16  # Audio format (16-bit resolution)
#     CHANNELS = 1              # Mono channel
#     RATE = 44100              # Sampling rate (44.1kHz)
#     CHUNK = 1024              # Number of samples per chunk
#     THRESHOLD = 500  # Adjust this value based on your environment

#     # Initialize PyAudio object
#     p = pyaudio.PyAudio()

#     # Open stream
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK)

#     print("Listening for sound...")

#     try:
#         while True:
#             # Read audio data from the stream
#             data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

#             # Compute the average volume (amplitude)
#             volume = np.linalg.norm(data)

#             # Check if the volume exceeds the threshold
#             if volume > THRESHOLD:
#                 print("Sound detected!")
#                 notify_clients("Sound detected!")
#             else:
#                 notify_clients("No sound detected")
#             time.sleep(1)

#     except KeyboardInterrupt:
#         print("Program stopped.")

#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()

#     # Terminate PyAudio object
#     p.terminate()

# # Run the sound detection in a background thread
# sound_thread = threading.Thread(target=detect_sound)
# sound_thread.daemon = True  # Ensure it closes with the main program
# sound_thread.start()

# ============================ End Audio Detection ==============================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
