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
KEY_CODES = {'W': 17, 'A': 30, 'S': 31, 'D': 32, 'SPACE': 57, 'LSHIFT': 42, 'LCTRL': 29} 
key_states = {'W': False, 'A': False, 'S': False, 'D': False, 'SPACE': False, 'LSHIFT': False, 'LCTRL': False}
joy_center = None
left_scroll_anchor = None
last_scroll_tick_time = 0
left_scroll_start_time = 0

# --- Global State ---
gestures_enabled = True
STATUS_FILE = "/tmp/gesture_status.txt"

def update_waybar_status(enabled):
    """Writes the current state to a RAM file for Waybar to read"""
    status = "🟢 " if enabled else "🛑 "
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(status + "\n")
    except Exception as e:
        print(f"Failed to update Waybar: {e}")

# Initialize the file on startup
update_waybar_status(gestures_enabled)

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
last_toggle_time = 0  
fist_start_time = 0
trail = deque(maxlen=30)  
last_rofi_time = 0
peace_held = False
gojo_held = False

left_physical_down = False
left_is_holding = False
left_down_time = 0

right_physical_down = False
right_is_holding = False
right_down_time = 0

COOLDOWN = 1.0  
CLICK_COOLDOWN = 0.3  
ROFI_COOLDOWN = 2.0       
FIST_HOLD_TIME = 0.2  
SWIPE_Y_THRESHOLD = 0.08  
SWIPE_X_THRESHOLD = 0.15  
CLICK_THRESHOLD = 0.04    
TUCK_THRESHOLD = 0.08  
PIANO_THRESHOLD = 0.03

# 3. Hook into RGB camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while cap.isOpened():
        # --- Check for Waybar Click Trigger ---
        if os.path.exists("/tmp/gesture_toggle_cmd"):
            os.remove("/tmp/gesture_toggle_cmd") # Delete the trigger
            gestures_enabled = not gestures_enabled
            update_waybar_status(gestures_enabled)
            print(f"System State Changed (via Waybar): {gestures_enabled}")
            
            # Instantly release all held keys and reset states when disabled
            if not gestures_enabled:
                for k in key_states: update_key(k, False)
                joy_center = None
                pointer_prev_x = None
                trail.clear()
                
                # SAFETY CATCH: Release mouse buttons
                if left_is_holding:
                    os.system("ydotool click 0x80")
                    left_is_holding = False
                if right_is_holding:
                    os.system("ydotool click 0x81")
                    right_is_holding = False
                left_physical_down = False
                right_physical_down = False

        success, img = cap.read()
        if not success: break

        img = cv2.flip(img, 1) 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        h, w, c = img.shape
        split_x = int(w * 0.35)
        cv2.line(img, (split_x, 0), (split_x, h), (50, 50, 50), 2)

        # Visual indicator of the global state on the camera feed
        status_color = (0, 255, 0) if gestures_enabled else (0, 0, 255)
        cv2.putText(img, f"SYSTEM: {'ON' if gestures_enabled else 'OFF'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                is_left_hand = hand_landmarks[9].x < 0.35
                
                thumb_tip = hand_landmarks[4]
                index_base = hand_landmarks[5]
                pinky_base = hand_landmarks[17]
                
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
                
                ring_up_strict = ring_tip.y < ring_pip.y - 0.05 
                
                ring_down = not ring_up
                pinky_down = not pinky_up
                index_curled = not index_up
                middle_curled = not middle_up
                ring_curled = not ring_up
                pinky_curled = not pinky_up

                # ==========================================
                # LEFT HAND
                # ==========================================
                if is_left_hand:
                    
                    # Check for the two-finger up poses (System Toggles - ALWAYS ACTIVE)
                    if index_up and middle_up and ring_down and pinky_down:
                        finger_dist = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
                        
                        if finger_dist > 0.05:
                            # --- PEACE SIGN (Fingers Spread): Global Killswitch ---
                            gojo_held = False 
                            if not peace_held: 
                                gestures_enabled = not gestures_enabled
                                update_waybar_status(gestures_enabled)
                                print(f"System State Changed: {gestures_enabled}")
                                peace_held = True 
                                
                                if not gestures_enabled:
                                    for k in key_states: update_key(k, False)
                                    joy_center = None
                                    pointer_prev_x = None
                                    trail.clear()
                                    left_scroll_anchor = None
                                    
                                    if left_is_holding:
                                        os.system("ydotool click 0x80")
                                        left_is_holding = False
                                    if right_is_holding:
                                        os.system("ydotool click 0x81")
                                        right_is_holding = False
                                    
                            continue 
                        else:
                            # --- DOMAIN EXPANSION: Hexecute Mode ---
                            peace_held = False 
                            if not gojo_held:
                                print("DOMAIN EXPANSION: Ready to cast!")
                                gojo_held = True 
                            continue 
                    else:
                        peace_held = False
                        gojo_held = False 

                    # =======================================================
                    # ONLY RUN THESE LEFT-HAND GESTURES IF SYSTEM IS ENABLED
                    # =======================================================
                    if gestures_enabled and not peace_held and not gojo_held:
                        
                        # --- LEFT HAND SCROLL JOYSTICK (Index Finger Only) ---
                        if index_up and middle_curled and ring_curled and pinky_curled:
                            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                            
                            # Phase 1: The Settling Time (0.3 seconds)
                            if left_scroll_anchor is None:
                                if left_scroll_start_time == 0:
                                    left_scroll_start_time = current_time
                                
                                # Draw a grey circle to show it is "loading" the anchor
                                cv2.circle(img, (ix, iy), 10, (150, 150, 150), cv2.FILLED)
                                
                                # Lock the anchor once the time has passed
                                if current_time - left_scroll_start_time > 0.3:
                                    left_scroll_anchor = iy
                            
                            # Phase 2: The Anchor is Locked, Execute Scrolling
                            else:
                                # Draw the Deadzone Ring and an "Elastic Band" line
                                cv2.circle(img, (ix, left_scroll_anchor), 20, (255, 255, 255), 2) 
                                cv2.line(img, (ix, left_scroll_anchor), (ix, iy), (255, 105, 180), 4)
                                cv2.circle(img, (ix, iy), 15, (255, 105, 180), cv2.FILLED)
                                
                                delta_y = iy - left_scroll_anchor
                                SCROLL_DEADZONE = 20 
                                
                                if abs(delta_y) > SCROLL_DEADZONE:
                                    speed_factor = min(1.0, (abs(delta_y) - SCROLL_DEADZONE) / 80.0)
                                    scroll_delay = 0.3 - (speed_factor * 0.25) 
                                    
                                    if current_time - last_scroll_tick_time > scroll_delay:
                                        if delta_y > 0:
                                            os.system("ydotool mousemove -w -- 0 -1") # Scroll Down
                                        else:
                                            os.system("ydotool mousemove -w -- 0 1")  # Scroll Up
                                        last_scroll_tick_time = current_time
                                        
                            continue # Skip WASD logic while scrolling
                        
                        # Reset everything if the finger drops
                        else:
                            left_scroll_anchor = None 
                            left_scroll_start_time = 0
                        
                        # --- WASD JOYSTICK (All fingers up) ---
                        if index_up and middle_up and ring_up and pinky_up:
                            palm_x, palm_y = hand_landmarks[9].x, hand_landmarks[9].y
                            if joy_center is None: joy_center = (palm_x, palm_y)
                            
                            cx, cy = int(joy_center[0] * w), int(joy_center[1] * h)
                            cv2.circle(img, (cx, cy), int(JOY_DEADZONE * w), (255, 255, 255), 2)
                            cv2.circle(img, (int(palm_x * w), int(palm_y * h)), 15, (0, 255, 0), cv2.FILLED)
                            
                            update_key('W', palm_y < joy_center[1] - JOY_DEADZONE)
                            update_key('S', palm_y > joy_center[1] + JOY_DEADZONE)
                            update_key('A', palm_x < joy_center[0] - JOY_DEADZONE)
                            update_key('D', palm_x > joy_center[0] + JOY_DEADZONE)

                            # --- THE THUMB PIANO (Shift, Ctrl, Space) ---
                            index_knuckle = hand_landmarks[5]
                            thumb_base = hand_landmarks[2] # Move Ctrl away from the index finger
                            pinky_knuckle = hand_landmarks[17]
                            
                            dist_shift = math.hypot(thumb_tip.x - index_knuckle.x, thumb_tip.y - index_knuckle.y)
                            # Ctrl is now triggered by curling your thumb straight down to its own base joint
                            dist_ctrl = math.hypot(thumb_tip.x - thumb_base.x, thumb_tip.y - thumb_base.y) 
                            dist_space = math.hypot(thumb_tip.x - pinky_knuckle.x, thumb_tip.y - pinky_knuckle.y)
                            
                            # Find the absolute closest anchor to the thumb
                            closest_dist = min(dist_shift, dist_ctrl, dist_space)
                            
                            # Only activate IF it's the closest key AND it's inside the tighter piano radius
                            shift_active = (closest_dist == dist_shift) and (dist_shift < PIANO_THRESHOLD)
                            ctrl_active = (closest_dist == dist_ctrl) and (dist_ctrl < PIANO_THRESHOLD)
                            space_active = (closest_dist == dist_space) and (dist_space < PIANO_THRESHOLD)
                            
                            update_key('LSHIFT', shift_active)
                            update_key('LCTRL', ctrl_active)
                            update_key('SPACE', space_active)
                            
                            # Visual Feedback
                            if shift_active:
                                cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (255, 100, 0), cv2.FILLED)
                                cv2.putText(img, "SHIFT", (cx - 50, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                            elif ctrl_active:
                                cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 100, 255), cv2.FILLED)
                                cv2.putText(img, "CTRL", (cx - 50, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                            elif space_active:
                                cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (255, 255, 0), cv2.FILLED)
                                cv2.putText(img, "SPACE", (cx - 50, cy - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                        # --- SAFETY RESET ---
                        else:
                            if joy_center is not None:
                                for k in key_states: update_key(k, False)
                                joy_center = None

                # ==========================================
                # RIGHT HAND (Only process if enabled)
                # ==========================================
                elif gestures_enabled:
                    if index_up and middle_curled and ring_curled and pinky_curled:
                        fist_start_time = 0 
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        cv2.circle(img, (ix, iy), 15, (0, 255, 255), cv2.FILLED)
                        
                        trail.append((ix, iy))
                        for i in range(1, len(trail)): cv2.line(img, trail[i-1], trail[i], (0, 255, 255), 3)

                        if len(trail) == 30 and (current_time - last_rofi_time > ROFI_COOLDOWN):
                            min_x, max_x = min([p[0] for p in trail]), max([p[0] for p in trail])
                            min_y, max_y = min([p[1] for p in trail]), max([p[1] for p in trail])
                            box_w, box_h = max_x - min_x, max_y - min_y
                            
                            if box_w > 80 and box_h > 80 and 0.7 < (box_w / box_h) < 1.3:
                                if math.hypot(trail[0][0] - trail[-1][0], trail[0][1] - trail[-1][1]) < 60:
                                    os.system("rofi -show drun -show-icons -theme ~/.config/rofi/gesture.rasi &")
                                    last_rofi_time = current_time
                                    trail.clear()  

                        dist_left = math.hypot(thumb_tip.x - index_base.x, thumb_tip.y - index_base.y)
                        dist_right = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)

                        # --- LEFT CLICK / DRAG AND DROP (Tap vs Hold) ---
                        if dist_left < CLICK_THRESHOLD and dist_left < dist_right:
                            cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 0, 255), cv2.FILLED)
                            if not left_physical_down:
                                left_physical_down = True
                                left_down_time = current_time
                            elif not left_is_holding and (current_time - left_down_time > 0.5):
                                os.system("ydotool click 0x40") # Convert to Hold/Drag
                                left_is_holding = True
                        else:
                            if left_physical_down:
                                left_physical_down = False
                                if left_is_holding:
                                    os.system("ydotool click 0x80") # Release Drag
                                    left_is_holding = False
                                else:
                                    if current_time - last_click_time > CLICK_COOLDOWN:
                                        os.system("ydotool click 0xC0") # Clean Atomic Click
                                        last_click_time = current_time

                        # --- RIGHT CLICK / HOLD (Tap vs Hold) ---
                        if dist_right < CLICK_THRESHOLD:
                            cv2.circle(img, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 20, (0, 255, 0), cv2.FILLED)
                            if not right_physical_down:
                                right_physical_down = True
                                right_down_time = current_time
                            elif not right_is_holding and (current_time - right_down_time > 0.5):
                                os.system("ydotool click 0x41") # Convert to Right Hold
                                right_is_holding = True
                        else:
                            if right_physical_down:
                                right_physical_down = False
                                if right_is_holding:
                                    os.system("ydotool click 0x81") # Release Right Hold
                                    right_is_holding = False
                                else:
                                    if current_time - last_click_time > CLICK_COOLDOWN:
                                        os.system("ydotool click 0xC1") # Clean Atomic Right Click
                                        last_click_time = current_time

                        if pointer_prev_x is None: pointer_prev_x, pointer_prev_y = ix, iy
                        move_x, move_y = (ix - pointer_prev_x) * SENSITIVITY, (iy - pointer_prev_y) * SENSITIVITY
                        
                        if abs(move_x) > DEADZONE or abs(move_y) > DEADZONE:
                            os.system(f"ydotool mousemove -- {int(move_x)} {int(move_y)}")
                        pointer_prev_x, pointer_prev_y = ix, iy

                    elif index_curled and middle_curled and ring_curled and pinky_curled and thumb_tip.y < index_base.y - 0.08:
                        pointer_prev_x = None 
                        trail.clear()
                        if current_time - last_enter_time > COOLDOWN:
                            os.system("ydotool key 28:1 28:0")
                            last_enter_time = current_time

                    elif index_curled and middle_curled and ring_curled and pinky_up:
                        pointer_prev_x = None 
                        trail.clear()
                        if current_time - last_esc_time > COOLDOWN:
                            os.system("ydotool key 1:1 1:0") 
                            last_esc_time = current_time

                    elif index_up and middle_up and ring_up_strict and pinky_down:
                        pointer_prev_x = None 
                        trail.clear()
                        avg_y = (index_tip.y + middle_tip.y + ring_tip.y) / 3
                        if 'close_prev_y' in locals() and (current_time - last_close_time > COOLDOWN):
                            if avg_y - close_prev_y > SWIPE_Y_THRESHOLD:
                                os.system("hyprctl dispatch killactive")
                                last_close_time = current_time
                                close_prev_y = avg_y
                                continue
                        close_prev_y = avg_y

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
                        
                        # SAFETY CATCH: Release mouse buttons
                        if left_is_holding:
                            os.system("ydotool click 0x80")
                            left_is_holding = False
                        if right_is_holding:
                            os.system("ydotool click 0x81")
                            right_is_holding = False
                        left_physical_down = False
                        right_physical_down = False

        cv2.imshow("Greasy Hands Protocol", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

finally:
    # ==========================================
    # THE ULTIMATE FAILSAFE (Runs when script dies)
    # ==========================================
    print("\n[!] Script terminating. Executing emergency key release...")
    
    # 1. Release all WASD and Modifier keys
    for key in key_states:
        os.system(f"ydotool key {KEY_CODES[key]}:0")
        
    # 2. Release all mouse buttons
    os.system("ydotool click 0x80") # Left click UP
    os.system("ydotool click 0x81") # Right click UP
    
    # 3. Turn off the Waybar status
    update_waybar_status(False)
    
    # 4. Release the camera
    cap.release()
    cv2.destroyAllWindows()
    print("[!] All virtual inputs successfully released.")
