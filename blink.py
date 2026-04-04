import argparse
import math
import time

import cv2
import mediapipe as mp


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))

    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])

    if horizontal <= 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def draw_eye_points(frame, landmarks, eye_indices, frame_w, frame_h):
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to face_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--closed-threshold", type=float, default=0.15,
                        help="EAR below this means eye is closed")
    parser.add_argument("--open-threshold", type=float, default=0.20,
                        help="EAR above this means eye is open again")
    parser.add_argument("--width", type=int, default=640,
                        help="Inference/display width")
    parser.add_argument("--height", type=int, default=480,
                        help="Inference/display height")
    parser.add_argument("--min-face-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-face-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    if args.open_threshold <= args.closed_threshold:
        raise ValueError("--open-threshold must be greater than --closed-threshold")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_face_presence_confidence=args.min_face_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(args.camera)

    # Reduce lag from camera buffering where supported.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    blink_count = 0
    eye_state = "open"   # states: "open", "closed"
    avg_ear = 0.0
    fps = 0.0
    last_loop_t = time.perf_counter()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)

            # Optional resize for speed consistency
            frame_bgr = cv2.resize(frame_bgr, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame_rgb.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Monotonic timestamp in ms is required for VIDEO mode.
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status_text = "No face"

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, frame_w, frame_h)
                right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, frame_w, frame_h)
                avg_ear = (left_ear + right_ear) / 2.0

                draw_eye_points(frame_bgr, face_landmarks, LEFT_EYE_IDX, frame_w, frame_h)
                draw_eye_points(frame_bgr, face_landmarks, RIGHT_EYE_IDX, frame_w, frame_h)

                # Hysteresis-based state machine:
                # open -> closed when EAR drops below closed threshold
                # closed -> open when EAR rises above open threshold
                if eye_state == "open":
                    status_text = "Eyes open"
                    if avg_ear < args.closed_threshold:
                        eye_state = "closed"
                        status_text = "Eyes closed"
                else:
                    status_text = "Eyes closed"
                    if avg_ear > args.open_threshold:
                        eye_state = "open"
                        blink_count += 1
                        status_text = "Blink counted"

            now = time.perf_counter()
            dt = now - last_loop_t
            last_loop_t = now
            if dt > 0:
                fps = 1.0 / dt

            cv2.putText(frame_bgr, f"Blinks: {blink_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"EAR: {avg_ear:.3f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame_bgr, f"State: {eye_state}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(
                frame_bgr,
                f"closed<{args.closed_threshold:.3f} open>{args.open_threshold:.3f}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
            )
            cv2.putText(frame_bgr, status_text, (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 255, 180), 2)

            cv2.imshow("Fast Blink Counter", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()