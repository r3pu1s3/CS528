'''
This file is to retrieve and process frames to retrieve landmark and calculate gaze
'''
import numpy as np, mediapipe as mp
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import math



# here is link for face: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# here is link for specific region: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

# these are indices of the iris 
RIGHT_IRIS  = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]



# for gaze
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [263, 362, 386, 374]

# for blink
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

'''
converts CV frame to mediapipe object
'''
def to_mp_img(frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

'''
function to start streaming and return frames 
input - maybe specific camera?
output - camera object
'''      

def initialize_camera(cam=0):
    # initialize camera
    cap = cv2.VideoCapture(cam)
    return cap

'''
gets frame 
input - the camera object
output - frame object
'''
def get_frame(cap):
    # returns safety bool, camera
    ret, frame = cap.read()
    
    # return error 0 if camera not accepting
    if not ret:
        return None
    
    frame = cv2.flip(frame, 1)
    return frame


'''
retrieves the detector in set config
'''
def init_detector():
    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="face_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )
    return detector

'''
gets coordinate for a point on the mech in the specific image
input - lm: mesh array, i: indices of that array, w: width of frame, h: height of frame
output - (x,y) array float 32
'''
def pt(lm, i, w, h):
    # lm[i].x or y gets relative position from 0 to 1
    return np.array([lm[i].x*w, lm[i].y*h], np.float32)

'''
runs detector and retrieves mesh array
input:
detector: mediapipe detector object
frame: the individual frame
output:
error or mesh array lm
'''
def run_detector(detector, image):
    res = detector.detect(image)
    if not res.face_landmarks:
        return None
    return res.face_landmarks[0]

'''
draws a dot on the frame based on one of the indicies in the mesh
'''
def draw_eye_points(frame, lm, eye_indices, w, h):
    for idx in eye_indices:
        
        x, y = pt(lm, idx, w, h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
'''
-------------------------------------
gaze:
'''

'''
gets the coordinate of the iris relative to the frame
'''
def get_iris_coord(lm, w, h, iris_ids):
    return np.mean([pt(lm, i, w, h) for i in iris_ids], axis=0)

'''
gets 4 coordinate for the eyes -> for gaze. may not be needed
'''
def get_eye_coord(lm, w, h, region_ids):
    out_i, in_i, top_i, bot_i = region_ids
    return pt(lm, out_i, w, h), pt(lm, in_i, w, h), pt(lm, top_i, w, h), pt(lm, bot_i, w, h)
'''
retrieves iris relative position to the eye region of a single eye
input-
lm: mesh array
iris_ids: indices for iris
_i: indices for eye region
w: width of iris
h: height of iris
output:
nx: relative x position in eye region
ny: relative y position in y region 
'''
def eye_xy(lm, iris_ids, region_ids, w, h):
    # calculates center of iris (x,y) coord
    iris = get_iris_coord(lm, w,h, iris_ids)
    
    # gets (x,y) for each landmark of the eye region as defined above
   
    outp, inp, topp, botp = get_eye_coord(lm, w, h, region_ids)
    # calculates the total width and height of the eye
    x_axis = inp - outp
    y_axis = botp - topp

    # calculates relative position of iris relative to eye region
    nx = np.dot(iris - outp, x_axis) / (np.dot(x_axis, x_axis) + 1e-6)
    ny = np.dot(iris - topp, y_axis) / (np.dot(y_axis, y_axis) + 1e-6)
    
    return nx, ny


'''
averages relative iris pos of both eyes and return general gaze
'''

def gaze_xy(lm, h, w):
    # h, w = frame.shape[:2]
    
    # runs detector on the frame
    # lm = run_detector(detector, frame)

    # gets left relative iris pos
    lnx, lny = eye_xy(lm, LEFT_IRIS,  LEFT_EYE, w, h)
    
    # gets right relative iris pos
    rnx, rny = eye_xy(lm, RIGHT_IRIS, RIGHT_EYE, w, h)

    # averages position - i.e. gets general gaze of both eyes
    nx = (lnx + rnx) / 2.0
    ny = (lny + rny) / 2.0
    return nx, ny
'''
--------------------------------------------------------------------
blinking:
'''



'''
calculates euclidean distance 
'''

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

'''
calculates EAR i.e. how wide the eyes are open

'''
def eye_aspect_ratio(lm, eye_indices, w, h):
    # gets respective point on the grid -6 in total
    pts = []
    for idx in eye_indices:
        pts.append(pt(lm, idx, w, h))
    # calculate vertical height of right and left of center
    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    
    # calculates horizontal 
    horizontal = euclidean(pts[0], pts[3])

    
    # note could raise div by zero error
    if horizontal <= 1e-6:
        return 0.0

    # returns EAR ratio
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

