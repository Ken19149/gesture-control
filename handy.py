import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
from collections import deque  # Added to track the finger path

# --- Trackpad Configuration ---
SENSITIVITY = 3.5  
DEADZONE = 2.0     

# 1. Model setup
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# 2. State Tracking Variables
pointer_prev_x, pointer_prev_y = None, None
last_waybar_time = 0
last_workspace_time = 0
last_click_time = 0
fist_start_time = 0

# --- New Gesture Variables ---
trail = deque(maxlen=30)  # Remembers the last 30 positions of your index finger
last_rofi_time = 0

COOLDOWN = 1.0  
CLICK_COOLDOWN = 0.3  
ROFI_COOLDOWN = 2.0       # Prevent rofi from spawning 5 times in a row
FIST_HOLD_TIME = 0.2  
SWIPE_Y_THRESHOLD = 0.08  
SWIPE_X_THRESHOLD = 0.15  
CLICK_THRESHOLD = 0.04    

# 3. Hook into RGB camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) 
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            thumb_tip = hand_landmarks[4]
            index_base = hand_landmarks[5]
            
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

            # --- GESTURE 1: POINTER MODE, CLICKING & CIRCLES ---
            if index_up and not middle_up and ring_down and pinky_down:
                fist_start_time = 0 
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(img, (ix, iy), 15, (0, 255, 255), cv2.FILLED)
                
                # --- The "Ink" Trail Logic ---
                trail.append((ix, iy))
                
                # Draw the glowing path
                for i in range(1, len(trail)):
                    cv2.line(img, trail[i-1], trail[i], (0, 255, 255), 3)

                # --- The Circle Detection Math ---
                if len(trail) == 30 and (current_time - last_rofi_time > ROFI_COOLDOWN):
                    # Find the bounding box of your drawing
                    min_x = min([p[0] for p in trail])
                    max_x = max([p[0] for p in trail])
                    min_y = min([p[1] for p in trail])
                    max_y = max([p[1] for p in trail])
                    
                    box_w = max_x - min_x
                    box_h = max_y - min_y
                    
                    # 1. Is the drawing big enough? (At least 80x80 camera pixels)
                    if box_w > 80 and box_h > 80:
                        # 2. Is it roughly a square shape (not a long line)?
                        if 0.7 < (box_w / box_h) < 1.3:
                            # 3. Did you close the loop? (Distance from start of trail to end of trail)
                            start_p = trail[0]
                            end_p = trail[-1]
                            dist_closed = math.hypot(start_p[0] - end_p[0], start_p[1] - end_p[1])
                            
                            if dist_closed < 60:
                                print("CIRCLE DETECTED! Launching Rofi...")
                                # Launches Rofi with a massive, centered 3x3 icon grid
                                os.system("rofi -show drun -show-icons -theme ~/.config/rofi/gesture.rasi &")
                                last_rofi_time = current_time
                                trail.clear()  # Erase the ink so it doesn't double-trigger

                # --- The Thumb Click Logic ---
                thumb_dist = math.hypot(thumb_tip.x - index_base.x, thumb_tip.y - index_base.y)
                if thumb_dist < CLICK_THRESHOLD:
                    cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 0, 255), cv2.FILLED)
                    if current_time - last_click_time > CLICK_COOLDOWN:
                        os.system("ydotool click 0xC0")
                        last_click_time = current_time

                # --- The Mouse Movement Logic ---
                if pointer_prev_x is None:
                    pointer_prev_x, pointer_prev_y = ix, iy
                
                delta_x = ix - pointer_prev_x
                delta_y = iy - pointer_prev_y
                
                move_x = delta_x * SENSITIVITY
                move_y = delta_y * SENSITIVITY
                
                if abs(move_x) > DEADZONE or abs(move_y) > DEADZONE:
                    os.system(f"ydotool mousemove -- {int(move_x)} {int(move_y)}")
                
                pointer_prev_x, pointer_prev_y = ix, iy

            # --- GESTURE 2: WAYBAR TOGGLE ---
            elif index_up and middle_up and ring_down and pinky_down:
                pointer_prev_x = None 
                fist_start_time = 0 
                trail.clear() # Erase ink if gesture changes
                
                cv2.circle(img, (int(index_tip.x * w), int(index_tip.y * h)), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (int(middle_tip.x * w), int(middle_tip.y * h)), 10, (255, 0, 0), cv2.FILLED)
                
                avg_y = (index_tip.y + middle_tip.y) / 2
                if 'waybar_prev_y' in locals() and (current_time - last_waybar_time > COOLDOWN):
                    delta_y = avg_y - waybar_prev_y
                    if abs(delta_y) > SWIPE_Y_THRESHOLD:
                        os.system("killall -SIGUSR1 waybar")
                        last_waybar_time = current_time
                        waybar_prev_y = avg_y
                        continue
                waybar_prev_y = avg_y

            # --- GESTURE 3: WORKSPACE GRAB ---
            elif index_curled and middle_curled and ring_down and pinky_down:
                pointer_prev_x = None 
                trail.clear() # Erase ink if gesture changes
                
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
                pointer_prev_x = None
                trail.clear() # Erase ink if you put your hand down

    cv2.imshow("Greasy Hands Protocol", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
