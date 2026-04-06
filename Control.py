import CV
import time
import serial
import numpy as np
import cv2
from Gaze_Classifier import Gaze_Classifier
from collections import deque


# Note for MAC command to activate virtual python environment is 
# source .venv/bin/activate 
# Additionally, MAC serial port is 

ser = serial.Serial('/dev/tty.usbserial-1140', 115200)


'''
Compiles all the tools from CV to create UI/control loop for the user
'''

def menu():
    while (True):
        pass

def config(t, deadzone, confirm_frames):
    xs, ys = [], []

    # initialize camera, detector
    cap = CV.initialize_camera(0)

    detector = CV.init_detector()
    
    # get time
    t0 = time.time()
    # run for 4 seconds
    while time.time() - t0 < t:
        # get and process frame, return gaze
        frame = CV.get_frame(cap)
        h,w = frame.shape[:2]

        img = CV.to_mp_img(frame)

        lm = CV.run_detector(detector, img)
        
        # sample center gaze
        if lm:
            g = CV.gaze_xy(lm, h, w)
            xs.append(g[0]); ys.append(g[1])
                
        cv2.waitKey(1) 
        cv2.imshow("Feed", frame)


    if xs and ys:
        cx, cy = float(np.median(xs)), float(np.median(ys))
        print("center:", cx, cy)
        # eval_js(f"setStatus('calibrated center: {cx:.3f}, {cy:.3f}')")
        cv2.destroyAllWindows()

        return Gaze_Classifier(cx, cy, deadzone, confirm_frames)
    else:
        print("Failed to calibrate center. Ensure webcam is active.")


# does live tracking
def live_tracking(gc:Gaze_Classifier, closed_threshold = 0.15, open_threshold = 0.2, gesture_window = 3, gesture_cooldown=1.5):
    # init camera and detector
    cap = CV.initialize_camera(0)
    detector = CV.init_detector()
    
    # state variables
    blink_count = 0
    eye_state = "open"   # states: "open", "closed"
    avg_ear = 0.0
    
    signal_active = False
    blink_times = deque()
    last_gesture_time = -1e9
    last_signal_text = "DEACTIVATED"
    # streams
    while(True):
        
        # gets and process image
        frame = CV.get_frame(cap)
        img = CV.to_mp_img(frame)
        
        # gets height and width
        h,w = frame.shape[:2]
        
        # get face mesh
        lm = CV.run_detector(detector, img)
        
        
        
        # if mesh  is successfully retrieved
        if lm:
            # get time
            now = time.monotonic()

            # gets open or closed status:
            left_ear = CV.eye_aspect_ratio(lm, CV.LEFT_EYE_IDX, w, h)
            right_ear = CV.eye_aspect_ratio(lm, CV.RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if eye_state == "open":
                status_text = "Eyes open"
                if avg_ear < closed_threshold:
                    eye_state = "closed"
                    # status_text = "Eyes closed"
            else:
                status_text = "Eyes closed"
                if avg_ear > open_threshold:
                    eye_state = "open"
                    blink_count += 1
                    # note the time the user blinked
                    blink_times.append(now)

                    # status_text = "Blink counted"
            
            # only keep track of recent blinks
            while blink_times and (now - blink_times[0] > gesture_window):
                blink_times.popleft()

            # Trigger only if not in cooldown
            if now - last_gesture_time >= gesture_cooldown:
                if len(blink_times) >= 3:
                    signal_active = not signal_active
                    last_gesture_time = now
                    blink_times.clear()

                    if signal_active:
                        last_signal_text = "ACTIVATED"
                        print("ACTIVATED")
                    else:
                        last_signal_text = "DEACTIVATED"
                        print("DEACTIVATED")

            # print(blink_count)
            cv2.putText(frame, f"{last_signal_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # if activated, track gaze
            if signal_active:
                nx, ny = CV.gaze_xy(lm, h, w)
                

                label, _ = gc.update(nx, ny)
                # print(f"{nx}| {gc.cx} | {gc.history}")

        
                if label == "RIGHT":
                    '''
                    ROBOT COMMAND CODE FOR RIGHT
                    '''
                    ser.write(b'd')
                    
                elif label == "LEFT":
                    '''             
                    ROBOT COMMAND CODE FOR LEFT
                    '''
                    ser.write(b'a')
                    
                else:
                    '''
                    ROBOT COMMAND CODE FOR CENTER
                    '''
                    ser.write(b' ')
            
                print(label)
                cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # show image
        cv2.imshow("Gaze Direction", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1ms, listen for keypress
            break


# takes t seconds to config center for gaze
t=4
# a threshold to recognize right and left gaze
deadzone = 0.008
# number of frames to sample gaze
frames = 3

gc = config(t, deadzone, frames)

# live tracking based on coordinate stored in gc gaze classifier
live_tracking(gc)



