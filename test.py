import cv2 as cv
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Form, Request, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import pyaudio
import threading
import time
import io
import base64  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import httpx
import time 
from pydantic import BaseModel
from datetime import datetime
import json
from ultralytics import YOLO 
from PIL import Image


app = FastAPI()

# For templates (HTML rendering)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Fake in-memory user storage
fake_user_db = {
    "user1": {"username": "user1", "password": "password1"},
    "user2": {"username": "user2", "password": "password2"},
    "user3": {"username": "user3", "password": "password3"},
    "user4": {"username": "user4", "password": "password4"},
    "user5": {"username": "user5", "password": "password5"},
    "admin": {"username": "admin", "password": "adminpass"}
}
current_user = None

user_cheating_data = {user: [] for user in fake_user_db if user != "admin"}

# Face tracking setup
mp_face_mesh = mp.solutions.face_mesh

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

eye_cheating = False
head_cheating = False
sound_detected = False
video_feed_active = False
video_cap = None

test_duration = 30  # 5 minutes in seconds
start_time = None
cheating_data = []

yolo_model = YOLO('yolov8m.pt')
class_names = ['person', 'book', 'cell phone']
class_indices = [i for i, name in yolo_model.names.items() if name in class_names]

multiple_persons_detected = False
book_detected = False
phone_detected = False

def detect_objects(frame):
    global multiple_persons_detected, book_detected, phone_detected
    
    results = yolo_model(frame, classes=class_indices)
    
    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            if class_name == 'person':
                person_count += 1
            elif class_name == 'book':
                book_detected = True
            elif class_name == 'cell phone':
                phone_detected = True
    
    multiple_persons_detected = person_count > 1

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
# audio_thread = threading.Thread(target=detect_sound)
# audio_thread.daemon = True  # Ensure it closes with the main program
# audio_thread.start()

def start_audio_detection():
    global audio_detection_active
    audio_detection_active = True
    audio_thread = threading.Thread(target=detect_sound)
    audio_thread.daemon = True  # Ensure it closes with the main program
    audio_thread.start()

def stop_audio_detection():
    global audio_detection_active
    audio_detection_active = False
    print("Audio detection stopped.")

anti_spoofing_model = YOLO('best.pt')

def predict_anti_spoofing(image):
    results = anti_spoofing_model(image, verbose=False)
    for result in results:
        probs = result.probs
        if probs.top1 == 1:
            return f"{anti_spoofing_model.names[probs.top1]}", probs.top1conf.item()
        elif probs.top1 == 0:
            return f"{anti_spoofing_model.names[probs.top1]}", probs.top1conf.item()
    return "Unknown", 0.0

def generate_video_feed(username):
    global eye_cheating, head_cheating, video_feed_active, multiple_persons_detected, book_detected, phone_detected
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while video_feed_active:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            detect_objects(frame)

            pil_image = Image.fromarray(frame_rgb)
            anti_spoofing_label, confidence = predict_anti_spoofing(pil_image)
            
            # Determine color based on the prediction
            color = (0, 255, 0) if anti_spoofing_label == "real" else (0, 0, 255)
            
            # Display anti-spoofing result on the frame
            cv.putText(frame, f"Anti-Spoofing: {anti_spoofing_label} ({confidence:.2f})", 
                       (20, 280), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
                eye_cheating = iris_pos != "CENTER"
                # cv.putText(frame, iris_pos, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                
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

                head_cheating = abs(y) > 10 or abs(x) > 10

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
                # cv.putText(frame, text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                cheating_votes = sum([
                    eye_cheating, 
                    head_cheating, 
                    sound_detected, 
                    multiple_persons_detected, 
                    book_detected, 
                    phone_detected,
                    anti_spoofing_label != "real"
                ])
                is_cheating = cheating_votes >= 2

                elapsed_time = time.time() - start_time
                user_cheating_data[username].append((elapsed_time, is_cheating))
                # cheating_data.append((elapsed_time, is_cheating))

                color = (0, 0, 255) if is_cheating else (0, 255, 0)
                text = "CHEATING DETECTED" if is_cheating else "NO CHEATING DETECTED"
                cv.putText(frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display individual module results
                cv.putText(frame, f"Eye: {'Cheating' if eye_cheating else 'OK'}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Head: {'Cheating' if head_cheating else 'OK'}", (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Sound: {'Detected' if sound_detected else 'Not Detected'}", (20, 160), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Multiple Persons: {'Detected' if multiple_persons_detected else 'Not Detected'}", (20, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Book: {'Detected' if book_detected else 'Not Detected'}", (20, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Phone: {'Detected' if phone_detected else 'Not Detected'}", (20, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            multiple_persons_detected = False
            book_detected = False
            phone_detected = False


    cap.release()


def generate_cheating_graph(username):
    if not user_cheating_data[username]:
        return None
    
    times, cheating = zip(*user_cheating_data[username])
    plt.figure(figsize=(10, 5))
    plt.plot(times, cheating, 'r-')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cheating Detected')
    plt.title(f'Cheating Instances Over Time for {username}')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['No', 'Yes'])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close('all')
    
    return image_base64

# @app.post("/alt-tab")
# async def alt_tab_alert(request: Request):
#     data = await request.json()
#     if data.get("alt_tab") == "true":
#         async with httpx.AsyncClient() as client:
#             response = await client.post("http://localhost:8000/alt-tab", json={"alt_tab": "true"})
#             return {"message": "Alt+Tab detected and reported"}
#     return {"message": "No Alt+Tab detected"}

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to render aboutUs.html
@app.get("/aboutUs", response_class=HTMLResponse)
async def get_about_us(request: Request):
    return templates.TemplateResponse("aboutUs.html", {"request": request})

# Route to render howItWorks.html
@app.get("/howItWorks", response_class=HTMLResponse)
async def get_how_it_works(request: Request):
    return templates.TemplateResponse("howItWorks.html", {"request": request})

# @app.get("/login", response_class=HTMLResponse)
# async def login_page(request: Request):
#     return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username in fake_user_db and password == fake_user_db[username]["password"]:
        global current_user
        current_user = username
        if username == "admin":
            return RedirectResponse(url="/admin", status_code=302)
        return RedirectResponse(url="/dashboard", status_code=302)
    return {"message": "Invalid credentials"}

# @app.post("/login")
# async def login(username: str = Form(...), password: str = Form(...)):
#     if username == fake_user_db["user"]["username"] and password == fake_user_db["user"]["password"]:
#         global current_user
#         current_user = username
#         return RedirectResponse(url="/dashboard", status_code=302)
#     return {"message": "Invalid credentials"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if current_user:
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})
    return RedirectResponse(url="/")

# @app.get("/admin", response_class=HTMLResponse)
# async def admin_dashboard(request: Request):
#     if current_user == "admin":
#         graph = generate_cheating_graph()
#         return templates.TemplateResponse("admin.html", {"request": request, "user": current_user, "graph": graph})
#     return RedirectResponse(url="/")

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    if current_user == "admin":
        users = [u for u in fake_user_db if u != "admin"]
        return templates.TemplateResponse("admin.html", {"request": request, "users": users})
    return RedirectResponse(url="/")

@app.get("/admin/user/{username}")
async def admin_user_dashboard(username: str):
    if current_user == "admin":
        graph = generate_cheating_graph(username)
        if graph:
            return HTMLResponse(f'<img src="data:image/png;base64,{graph}" alt="Cheating Graph for {username}">')
        return HTMLResponse('<p>No data available for this user.</p>')
    # raise HTTPException(status_code=403, detail="Not authorized")

@app.get("/video_feed/{username}")
async def video_feed(username: str):
    if current_user and video_feed_active:
        return StreamingResponse(generate_video_feed(username), media_type="multipart/x-mixed-replace; boundary=frame")
    return RedirectResponse(url="/")

@app.post("/alt-tab")
async def handle_alt_tab(request: Request):
    data = await request.json()
    type = data.get('type')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    time_elapsed = data.get('time_elapsed')

    # Convert timestamps to readable format
    start_time = datetime.fromtimestamp(start_time / 1000) if start_time else None
    end_time = datetime.fromtimestamp(end_time / 1000) if end_time else None

    # Log the received data
    print(f"Type: {type}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Time Elapsed: {time_elapsed}")

    return {"message": "Data received"}

@app.post("/toggle_video_feed")
async def toggle_video_feed():
    global video_feed_active, start_time, cheating_data, video_cap
    video_feed_active = not video_feed_active
    if video_feed_active:
        start_time = time.time()
        cheating_data = []
        start_audio_detection()  # Start audio detection along with video feed
    else:
        stop_audio_detection()  # Stop audio detection when video feed stops
        if video_cap is not None:
            video_cap.release()  # Release the camera
            video_cap = None  # Reset the capture object
    return {"status": "active" if video_feed_active else "inactive"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
