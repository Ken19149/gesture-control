import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time
import math
from collections import deque

# --- Trackpad Configuration ---
SENSITIVITY = 3.5  
DEADZONE = 2.0     

# --- WASD Joystick Configuration ---
JOY_DEADZONE = 0.05  
# Added SPACE (57), ENTER (28), and ESC (1) to Linux evdev keycodes
KEY_CODES = {'W': 17, 'A': 30, 'S': 31, 'D': 32, 'SPACE': 57} 
key_states = {'W': False, 'A': False, 'S': False, 'D': False, 'SPACE': False}
joy_center = None

def update_key(key, press):
    if key_states[key] != press:
        key_states[key] = press
        state = 1 if press else 0
        os.system(f"ydotool key {KEY_CODES[key]}:{state}")
        print(f"[{key}] {'DOWN' if press else 'UP'}")

# 1. Model setup 
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# 2. State Tracking Variables
pointer_prev_x, pointer_prev_y = None, None
last_waybar_time = 0
last_workspace_time = 0
last_click_time = 0
last_close_time = 0
last_enter_time = 0
last_esc_time = 0
fist_start_time = 0
trail = deque(maxlen=30)  
last_rofi_time = 0

COOLDOWN = 1.0  
CLICK_COOLDOWN = 0.3  
ROFI_COOLDOWN = 2.0       
FIST_HOLD_TIME = 0.2  
SWIPE_Y_THRESHOLD = 0.08  
SWIPE_X_THRESHOLD = 0.15  
CLICK_THRESHOLD = 0.04    
TUCK_THRESHOLD = 0.08  # Distance for thumb reaching across the palm

# 3. Hook into RGB camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    h, w, c = img.shape
    cv2.line(img, (w//2, 0), (w//2, h), (50, 50, 50), 2)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            is_left_hand = hand_landmarks[9].x < 0.5
            
            thumb_tip = hand_landmarks[4]
            index_base = hand_landmarks[5]
            pinky_base = hand_landmarks[17] # Used for Right Click and Spacebar
            
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]
            
            current_time = time.time()
            
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y
            ring_up = ring_tip.y < ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y
            ring_down = not ring_up
            pinky_down = not pinky_up
            index_curled = not index_up
            middle_curled = not middle_up
            ring_curled = not ring_up
            pinky_curled = not pinky_up

            # ==========================================
            # LEFT HAND: VIRTUAL WASD & SPACEBAR
            # ==========================================
            if is_left_hand:
                if index_up and middle_up and ring_up and pinky_up:
                    palm_x, palm_y = hand_landmarks[9].x, hand_landmarks[9].y
                    
                    if joy_center is None:
                        joy_center = (palm_x, palm_y)
                    
                    cx, cy = int(joy_center[0] * w), int(joy_center[1] * h)
                    cv2.circle(img, (cx, cy), int(JOY_DEADZONE * w), (255, 255, 255), 2)
                    cv2.circle(img, (int(palm_x * w), int(palm_y * h)), 15, (0, 255, 0), cv2.FILLED)
                    
                    update_key('W', palm_y < joy_center[1] - JOY_DEADZONE)
                    update_key('S', palm_y > joy_center[1] + JOY_DEADZONE)
                    update_key('A', palm_x < joy_center[0] - JOY_DEADZONE)
                    update_key('D', palm_x > joy_center[0] + JOY_DEADZONE)

                    # --- SPACEBAR (Jump): Thumb tucks across palm ---
                    thumb_tuck_dist = math.hypot(thumb_tip.x - pinky_base.x, thumb_tip.y - pinky_base.y)
                    update_key('SPACE', thumb_tuck_dist < TUCK_THRESHOLD)
                    
                    if thumb_tuck_dist < TUCK_THRESHOLD:
                        cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (255, 255, 0), cv2.FILLED)
                        cv2.putText(img, "JUMP!", (cx - 50, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                else:
                    if joy_center is not None:
                        for k in key_states: update_key(k, False)
                        joy_center = None

            # ==========================================
            # RIGHT HAND: MOUSE & GESTURES
            # ==========================================
            else:
                # --- POINTER MODE, LEFT/RIGHT CLICK, & CIRCLES ---
                if index_up and not middle_up and ring_down and pinky_down:
                    fist_start_time = 0 
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(img, (ix, iy), 15, (0, 255, 255), cv2.FILLED)
                    
                    trail.append((ix, iy))
                    for i in range(1, len(trail)):
                        cv2.line(img, trail[i-1], trail[i], (0, 255, 255), 3)

                    if len(trail) == 30 and (current_time - last_rofi_time > ROFI_COOLDOWN):
                        min_x, max_x = min([p[0] for p in trail]), max([p[0] for p in trail])
                        min_y, max_y = min([p[1] for p in trail]), max([p[1] for p in trail])
                        box_w, box_h = max_x - min_x, max_y - min_y
                        
                        if box_w > 80 and box_h > 80 and 0.7 < (box_w / box_h) < 1.3:
                            if math.hypot(trail[0][0] - trail[-1][0], trail[0][1] - trail[-1][1]) < 60:
                                os.system("rofi -show drun -show-icons -theme ~/.config/rofi/gesture.rasi &")
                                last_rofi_time = current_time
                                trail.clear()  

                    # Left Click (Thumb to Index Base)
                    if math.hypot(thumb_tip.x - index_base.x, thumb_tip.y - index_base.y) < CLICK_THRESHOLD:
                        cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 0, 255), cv2.FILLED)
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            os.system("ydotool click 0xC0")
                            last_click_time = current_time

                    # Right Click (Thumb to Middle PIP Knuckle)
                    if math.hypot(thumb_tip.x - middle_pip.x, thumb_tip.y - middle_pip.y) < CLICK_THRESHOLD:
                        cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 255, 0), cv2.FILLED)
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            print("RIGHT CLICK!")
                            os.system("ydotool click 0xC1") # 0xC1 is Right Click
                            last_click_time = current_time

                    # Mouse Movement
                    if pointer_prev_x is None:
                        pointer_prev_x, pointer_prev_y = ix, iy
                    
                    move_x = (ix - pointer_prev_x) * SENSITIVITY
                    move_y = (iy - pointer_prev_y) * SENSITIVITY
                    
                    if abs(move_x) > DEADZONE or abs(move_y) > DEADZONE:
                        os.system(f"ydotool mousemove -- {int(move_x)} {int(move_y)}")
                    pointer_prev_x, pointer_prev_y = ix, iy

                # --- ENTER (Thumbs Up) ---
                elif index_curled and middle_curled and ring_curled and pinky_curled and thumb_tip.y < index_base.y - 0.08:
                    pointer_prev_x = None 
                    trail.clear()
                    cv2.putText(img, "ENTER", (int(thumb_tip.x * w), int(thumb_tip.y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if current_time - last_enter_time > COOLDOWN:
                        os.system("ydotool key 28:1 28:0") # Taps Enter
                        last_enter_time = current_time

                # --- ESC (Pinky Up) ---
                elif index_curled and middle_curled and ring_curled and pinky_up:
                    pointer_prev_x = None 
                    trail.clear()
                    cv2.putText(img, "ESC", (int(pinky_tip.x * w), int(pinky_tip.y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if current_time - last_esc_time > COOLDOWN:
                        os.system("ydotool key 1:1 1:0") # Taps Esc
                        last_esc_time = current_time

                # --- CLOSE APP (3-Finger Swipe Down) ---
                elif index_up and middle_up and ring_up and pinky_down:
                    pointer_prev_x = None 
                    trail.clear()
                    
                    cv2.circle(img, (int(index_tip.x * w), int(index_tip.y * h)), 10, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (int(middle_tip.x * w), int(middle_tip.y * h)), 10, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (int(ring_tip.x * w), int(ring_tip.y * h)), 10, (0, 0, 255), cv2.FILLED)
                    
                    avg_y = (index_tip.y + middle_tip.y + ring_tip.y) / 3
                    
                    if 'close_prev_y' in locals() and (current_time - last_close_time > COOLDOWN):
                        if avg_y - close_prev_y > SWIPE_Y_THRESHOLD:
                            os.system("hyprctl dispatch killactive")
                            last_close_time = current_time
                            close_prev_y = avg_y
                            continue
                    close_prev_y = avg_y

                # --- WAYBAR TOGGLE ---
                elif index_up and middle_up and ring_down and pinky_down:
                    pointer_prev_x = None 
                    trail.clear()
                    
                    avg_y = (index_tip.y + middle_tip.y) / 2
                    if 'waybar_prev_y' in locals() and (current_time - last_waybar_time > COOLDOWN):
                        if abs(avg_y - waybar_prev_y) > SWIPE_Y_THRESHOLD:
                            os.system("killall -SIGUSR1 waybar")
                            last_waybar_time = current_time
                            waybar_prev_y = avg_y
                            continue
                    waybar_prev_y = avg_y

                # --- WORKSPACE GRAB ---
                elif index_curled and middle_curled and ring_down and pinky_down:
                    pointer_prev_x = None 
                    trail.clear()
                    
                    if fist_start_time == 0: fist_start_time = current_time
                    if current_time - fist_start_time > FIST_HOLD_TIME:
                        anchor = hand_landmarks[9]
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
                    trail.clear()

    cv2.imshow("Greasy Hands Protocol", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
