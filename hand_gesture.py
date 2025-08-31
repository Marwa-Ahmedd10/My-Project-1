import mediapipe as mp
import cv2
import pyautogui

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define landmark indices for finger tips
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# For scroll detection, tracking the previous Y-position of the index finger
prev_index_y = None

def classify_gesture(landmarks):
    """
    Determines the hand gesture based on key landmark positions.
    """
    global prev_index_y

    thumb_tip = landmarks[THUMB_TIP].y
    index_tip = landmarks[INDEX_TIP].y
    middle_tip = landmarks[MIDDLE_TIP].y
    ring_tip = landmarks[RING_TIP].y
    pinky_tip = landmarks[PINKY_TIP].y

    # Detect Open Hand (all fingers extended)
    if (index_tip < landmarks[INDEX_TIP - 2].y and
        middle_tip < landmarks[MIDDLE_TIP - 2].y and
        ring_tip < landmarks[RING_TIP - 2].y and
        pinky_tip < landmarks[PINKY_TIP - 2].y and
        thumb_tip < landmarks[THUMB_TIP - 2].y):  # Ensure the thumb is also extended
        return "Play/Pause"

    # Detect Fist (all fingers curled)
    elif (index_tip > landmarks[INDEX_TIP - 2].y and
          middle_tip > landmarks[MIDDLE_TIP - 2].y and
          ring_tip > landmarks[RING_TIP - 2].y and
          pinky_tip > landmarks[PINKY_TIP - 2].y and
          thumb_tip > landmarks[THUMB_TIP - 2].y):  # Ensure the thumb is also curled
        return "Mute/Unmute"

    # Detect Victory (index and middle fingers raised)
    elif (index_tip < landmarks[INDEX_TIP - 2].y and
          middle_tip < landmarks[MIDDLE_TIP - 2].y and
          ring_tip > landmarks[RING_TIP - 2].y and
          pinky_tip > landmarks[PINKY_TIP - 2].y and
          thumb_tip > landmarks[THUMB_TIP - 2].y):  # Ensure the thumb is curled
        return "Previous (Rewind)"

    # Detect Raise Index (only the index raised)
    elif index_tip < landmarks[INDEX_TIP - 2].y and \
         middle_tip > landmarks[MIDDLE_TIP - 2].y and \
         ring_tip > landmarks[RING_TIP - 2].y and \
         pinky_tip > landmarks[PINKY_TIP - 2].y:
        return "Next (Skip)"

    # Detect Pinky Up (Only pinky finger raised)
    elif pinky_tip < landmarks[PINKY_TIP - 2].y and \
         index_tip > landmarks[INDEX_TIP - 2].y and \
         middle_tip > landmarks[MIDDLE_TIP - 2].y and \
         ring_tip > landmarks[RING_TIP - 2].y:
        return "Increase Volume"

    # Detect Decrease Volume (index, middle, and ring fingers raised)
    elif index_tip < landmarks[INDEX_TIP - 2].y and \
         middle_tip < landmarks[MIDDLE_TIP - 2].y and \
         ring_tip < landmarks[RING_TIP - 2].y and \
         pinky_tip > landmarks[PINKY_TIP - 2].y:
        return "Decrease Volume"

    # Detect Scrolling (based on index finger movement)
    if prev_index_y is not None:
        if index_tip < prev_index_y - 0.05:
            prev_index_y = index_tip
            return "Next (Skip)"
        elif index_tip > prev_index_y + 0.05:
            prev_index_y = index_tip
            return "Scroll Down"
    prev_index_y = index_tip

    return "Unknown Gesture"

# For media control (volume adjustments)
def control_volume(action):
    if action == "increase":
        pyautogui.press("volumeup")  # Simulate volume up key press
    elif action == "decrease":
        pyautogui.press("volumedown")  # Simulate volume down key press