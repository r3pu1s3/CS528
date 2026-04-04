import CV
import cv2
import time
import numpy as np
'''
Compiles all the tools from CV to create UI/control loop for the user
'''



# testing
# cap = CV.initialize_camera(0)

# detector = CV.init_detector()
# # streams
# while(True):
    
#     # gets and process image
#     frame = CV.get_frame(cap)
#     img = CV.to_mp_img(frame)
    
#     # gets height and width
#     h,w = frame.shape[:2]
    
#     # get face mesh
#     lm = CV.run_detector(detector, img)
    
#     # get gaze
#     # nx, ny = None, None
#     if lm:
#     #     nx, ny = CV.gaze_xy(lm, h, w)
    
#         left_center = CV.get_iris_coord(lm, w, h, CV.LEFT_IRIS)
#         left_center = (int(left_center[0]), int(left_center[1]))
        
#         right_center = CV.get_iris_coord(lm, w, h, CV.RIGHT_IRIS)
#         right_center = (int(right_center[0]), int(right_center[1]))
#         cv2.circle(frame, left_center,  2, (0, 255, 0), -1)   # green ring
#         cv2.circle(frame, right_center,  2, (0, 255, 0), -1)   # green ring

    
#     # show image
#     cv2.imshow("Feed", frame)
    
#     # print(nx, ny)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1ms, listen for keypress
#         break







cx, cy = 0.5, 0.5
HX = 0.01
HY = 0.08
def menu():
    while (True):
        pass



def config():
    xs, ys = [], []

    # initialize camera, detector
    cap = CV.initialize_camera(0)

    detector = CV.init_detector()
    
    # get time
    t0 = time.time()
    global cx, cy
    # run for 4 seconds
    while time.time() - t0 < 4.0:
        # get and process frame, return gaze
        frame = CV.get_frame(cap)
        h,w = frame.shape[:2]

        img = CV.to_mp_img(frame)

        lm = CV.run_detector(detector, img)
        if lm:
            g = CV.gaze_xy(lm, h, w)
            # g = CV.eye_xy(lm, CV.RIGHT_IRIS, CV.RIGHT_EYE,  h, w)
            xs.append(g[0]); ys.append(g[1])
                
        cv2.waitKey(1) 
        cv2.imshow("Feed", frame)


    if xs and ys:
        cx, cy = float(np.median(xs)), float(np.median(ys))
        print("center:", cx, cy)
        # eval_js(f"setStatus('calibrated center: {cx:.3f}, {cy:.3f}')")
    else:
        print("Failed to calibrate center. Ensure webcam is active.")


# does live tracking
def live_tracking():
    # testing
    cap = CV.initialize_camera(0)

    detector = CV.init_detector()
    # streams
    while(True):
        
        # gets and process image
        frame = CV.get_frame(cap)
        img = CV.to_mp_img(frame)
        
        # gets height and width
        h,w = frame.shape[:2]
        
        
        # add sampling to get an overall gaze direction over a interval 
        # need to find a way to determine if user blinked
        
        # get face mesh
        lm = CV.run_detector(detector, img)
        
        # get gaze
        # nx, ny = None, None
        
        
        if lm:
            nx, ny = CV.gaze_xy(lm, h, w)
            # nx, ny =CV.eye_xy(lm, CV.RIGHT_IRIS, CV.RIGHT_EYE, h, w)
            left_center = CV.get_iris_coord(lm, w, h, CV.LEFT_IRIS)
            left_center = (int(left_center[0]), int(left_center[1]))
            
            right_center = CV.get_iris_coord(lm, w, h, CV.RIGHT_IRIS)
            right_center = (int(right_center[0]), int(right_center[1]))
            
            right_eye = CV.get_eye_coord(lm, w, h, CV.RIGHT_EYE)
            # left_eye = CV.get_eye_coord(lm, w, h, CV.LEFT_EYE)
            
            cv2.circle(frame, left_center,  2, (0, 255, 0), -1)   # green ring
            
            cv2.circle(frame, right_center,  2, (0, 255, 0), -1)   # green ring
            
            for point in right_eye:
                point = (int(point[0]), int(point[1]))
                cv2.circle(frame, point,  2, (0, 255, 0), -1)   # green ring

            dx, dy = nx - cx, ny - cy

            print(f"Pos:{nx}, {ny} | Center:{cx}, {cy} | Delta: {dx}, {dy} ")
            if dx < -HX:
                label = "LEFT"
            elif dx > HX:
                label = "RIGHT"
            else:
                label = "CENTER"
            print(label)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # eval_js(f"setStatus('gaze={label} | nx={nx:.3f} ny={ny:.3f} | dx={dx:.3f} dy={dy:.3f}')")
        # show image
        cv2.imshow("Feed", frame)
        
        # print(nx, ny)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # wait 1ms, listen for keypress
            break



config()
live_tracking()
'''
control loop: turn on/start (with gesture?) --> two option: 1. config 2. live tracking
--> config --> take 4 second to get center of gaze --> menu
--> live tracking --> track live --> until press a button/or do gesture to go back to menu
--> turn off with gesture

state within live tracking:
1. rotational tracking
2. lateral tracking

'''
