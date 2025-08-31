import cv2
import time
import numpy as np
from face_recognition import authenticate_face
from hand_gesture import classify_gesture, control_volume
from ocr import OCRModule
import mediapipe as mp
import pyautogui
import glob

# --------- Camera Calibration (Run Once) ---------
def calibrate_camera(calibration_folder='calibration_images'):
    chessboard_size = (9, 6)
    square_size = 1.0

    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(f"{calibration_folder}/*.png")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(refined)

    if len(objpoints) > 0:
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("[✅] Camera calibrated and saved.")
    else:
        print("[⚠️] No valid chessboard patterns found.")

# Run calibration once (comment out later if not needed again)
calibrate_camera()

# --------- Load Calibration Parameters ---------
try:
    data = np.load("calibration_data.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print("[✅] Calibration data loaded.")
except:
    camera_matrix = None
    dist_coeffs = None
    print("[⚠️] Calibration data not found. Continuing without undistortion.")

# --------- Smart Assistant Starts ---------
cap = cv2.VideoCapture(0)
authenticated = False
current_name = ""

ocr = OCRModule()
ocr.set_debug(False)
roi = (300, 200, 300, 100)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

print("📸 Press 'q' to quit")

frame_count = 0
gesture_text = ""
ocr_text = ""
last_action_time = 0
cooldown = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if camera_matrix is not None and dist_coeffs is not None:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Face Recognition every 30 frames
    frame_count += 1
    if frame_count % 30 == 0:
        resized_frame = cv2.resize(frame, (300, 300))
        name = authenticate_face(resized_frame)
        if name:
            current_name = name
            authenticated = True
        else:
            current_name = "Unknown"
            authenticated = False

    # 2. OCR
    ocr_text = ocr.extract_text(frame, roi)
    frame = ocr.draw_roi(frame, roi)

    # 3. Gesture Detection
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = classify_gesture(hand_landmarks.landmark)

            if authenticated:
                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    if gesture_text == "Play/Pause":
                        pyautogui.press("playpause")
                    elif gesture_text == "Mute/Unmute":
                        pyautogui.press("volumemute")
                    elif gesture_text == "Next (Skip)":
                        pyautogui.press('right')
                    elif gesture_text == "Previous (Rewind)":
                        pyautogui.press('left')
                    elif gesture_text == "Increase Volume":
                        control_volume("increase")
                    elif gesture_text == "Decrease Volume":
                        control_volume("decrease")
                    last_action_time = current_time

    # 4. UI Feedback
    auth_label = f"User: {current_name}" if authenticated else "Unauthorized"
    cv2.putText(frame, auth_label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if authenticated else (0, 0, 255), 2)
    cv2.putText(frame, f"OCR: {ocr_text[:30]}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_text}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)

    cv2.imshow("Smart Assistant", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
