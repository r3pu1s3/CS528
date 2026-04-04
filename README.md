# CS528
IoT Project for CS528 @ Umass. Automated wheelchair that tracks user’s eye movement both during the day and night to allow those with movement disabilities to accessibly navigate their physical surroundings. This project will try to test two methods for eye tracking: one with IR sensor and one with visible light. The user's forward gaze would be the signal to move forward and gaze left and right would signal turns. 


## Setup

1. Create and activate a virtual environment
2. Run `pip install -r requirements.txt`
3. Download the face landmark model:
   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
   Place it in the root of the project folder.