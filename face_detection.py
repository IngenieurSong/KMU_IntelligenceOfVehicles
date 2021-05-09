import cv2
import mediapipe as mp
import winsound as sd
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not results.detections:
            print(None)
            sd.Beep(2000, 2000)
            continue
        else:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                cv2.imshow('MediaPipe Face Detection', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

cap.release()