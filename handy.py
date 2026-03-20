import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

# --- Monitor Configuration ---
# Monitor 0: eDP-2 (Scaled to 1600x1000, starts at X=0)
# Monitor 1: HDMI-A-1 (1920x1080, starts at X=1600)
MONITORS = [
    {"w": 1600, "h": 1000, "offset_x": 0, "color": (0, 255, 255)},   # Yellow dot for Laptop
    {"w": 1920, "h": 1080, "offset_x": 1600, "color": (255, 255, 0)} # Cyan dot for HDMI
]
active_monitor = 0  # Start on the laptop screen

FRAME_MARGIN = 100  
SMOOTHING = 5       

# 1. Model setup
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# 2. State Tracking Variables
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

last_two_finger_time = 0
last_workspace_time = 0
fist_start_time = 0

COOLDOWN = 1.0  
FIST_HOLD_TIME = 0.2  
SWIPE_Y_THRESHOLD = 0.08  
SWIPE_X_THRESHOLD = 0.12  # Threshold for swapping monitors

# 3. Hook into RGB camera
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) 
    cv2.rectangle(img, (FRAME_MARGIN, FRAME_MARGIN), (cam_w - FRAME_MARGIN, cam_h - FRAME_MARGIN), (255, 0, 255), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]
            
            h, w, c = img.shape
            current_time = time.time()
            
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            index_curled = index_tip.y > index_pip.y
            middle_curled = middle_tip.y > middle_pip.y

            # --- GESTURE 1: POINTER MODE (Index UP, others DOWN) ---
            if index_up and not middle_up and ring_down and pinky_down:
                fist_start_time = 0 
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw dot based on active monitor color
                current_mon = MONITORS[active_monitor]
                cv2.circle(img, (ix, iy), 15, current_mon["color"], cv2.FILLED)
                
                # Map to the specific monitor's width/height
                mapped_x = int((ix - FRAME_MARGIN) / (cam_w - 2 * FRAME_MARGIN) * current_mon["w"])
                mapped_y = int((iy - FRAME_MARGIN) / (cam_h - 2 * FRAME_MARGIN) * current_mon["h"])
                
                # Clamp it so it can't escape the active monitor
                mapped_x = max(0, min(current_mon["w"], mapped_x))
                mapped_y = max(0, min(current_mon["h"], mapped_y))
                
                # Add the absolute offset for the Wayland canvas
                mapped_x += current_mon["offset_x"]
                
                curr_x = prev_x + (mapped_x - prev_x) / SMOOTHING
                curr_y = prev_y + (mapped_y - prev_y) / SMOOTHING
                
                os.system(f"ydotool mousemove --absolute {int(curr_x)} {int(curr_y)}")
                prev_x, prev_y = curr_x, curr_y

            # --- GESTURE 2: MULTI-TOOL (Index & Middle UP) ---
            elif index_up and middle_up and ring_down and pinky_down:
                fist_start_time = 0 
                avg_x = (index_tip.x + middle_tip.x) / 2
                avg_y = (index_tip.y + middle_tip.y) / 2
                
                # Draw blue tracking dots
                cv2.circle(img, (int(index_tip.x * w), int(index_tip.y * h)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(middle_tip.x * w), int(middle_tip.y * h)), 10, (255, 0, 0), cv2.FILLED)

                if 'two_finger_prev_x' in locals() and (current_time - last_two_finger_time > COOLDOWN):
                    delta_x = avg_x - two_finger_prev_x
                    delta_y = avg_y - two_finger_prev_y
                    
                    # Horizontal Swipes (Change Monitor)
                    if delta_x < -SWIPE_X_THRESHOLD:
                        print("SWAP TO LAPTOP")
                        active_monitor = 0
                        last_two_finger_time = current_time
                    elif delta_x > SWIPE_X_THRESHOLD:
                        print("SWAP TO HDMI")
                        active_monitor = 1
                        last_two_finger_time = current_time
                    
                    # Vertical Swipes (Waybar)
                    elif abs(delta_y) > SWIPE_Y_THRESHOLD:
                        os.system("killall -SIGUSR1 waybar")
                        last_two_finger_time = current_time
                        
                two_finger_prev_x = avg_x
                two_finger_prev_y = avg_y

            # --- GESTURE 3: WORKSPACE GRAB (All Curled) ---
            elif index_curled and middle_curled and ring_down and pinky_down:
                if fist_start_time == 0:
                    fist_start_time = current_time
                
                if current_time - fist_start_time > FIST_HOLD_TIME:
                    anchor = hand_landmarks[9]
                    cv2.circle(img, (int(anchor.x * w), int(anchor.y * h)), 15, (0, 255, 0), cv2.FILLED)

                    if 'grab_prev_x' in locals() and (current_time - last_workspace_time > COOLDOWN):
                        delta_x = anchor.x - grab_prev_x
                        if delta_x < -SWIPE_X_THRESHOLD:
                            os.system("hyprctl dispatch workspace e-1")
                            last_workspace_time = current_time
                        elif delta_x > SWIPE_X_THRESHOLD:
                            os.system("hyprctl dispatch workspace e+1")
                            last_workspace_time = current_time
                    grab_prev_x = anchor.x
            else:
                fist_start_time = 0

    cv2.imshow("Greasy Hands Protocol", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
