import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

ss_rosto = mp.solutions.face_detection
grph = mp.solutions.drawing_utils
rec_rosto = ss_rosto.FaceDetection()


while webcam.isOpened():
    verificador, frame = webcam.read()
    if not verificador:
        print('Webcam não detectada!')
        break
    
    img = frame
    ll_rostos = rec_rosto.process(img)

    if ll_rostos.detections:
        for rosto in ll_rostos.detections:
            grph.draw_detection(img, rosto)

    cv2.imshow("Face Detection", img)
    if cv2.waitKey(7) == 27:
        break

webcam.release()
cv2.destroyAllWindows()