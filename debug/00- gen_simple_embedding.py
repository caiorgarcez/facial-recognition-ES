import cv2
import face_recognition
import imutils

image = cv2.imread("./images/caio.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model="hog")
face_encoding = face_recognition.face_encodings(image, boxes)[0]
print(f"[INFO] Embedding vector:\n {face_encoding}\n\n")
print(f"[INFO] Length: {len(face_encoding)}")

with open("embedding.txt", "w", encoding="utf-8") as f:
    f.write(str(face_encoding))
