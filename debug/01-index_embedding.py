import cv2
import face_recognition
import imutils
import numpy
from elasticsearch import Elasticsearch

elastic_con = Elasticsearch([{"host": "localhost"}])

# TODO: replace with pickle
with open("embedding.txt", "r", encoding="utf-8") as f:
    face_encoding = f.read()

mapping = {
    "mappings": {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": 128},
            "user": {"type": "keyword"},
        }
    }
}

response = elastic_con.indices.create(index="users", ignore=400, body=mapping)
print(f"[INFO] ES index creation response: {response}")

image = cv2.imread("./images/caio.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb, model="hog")
face_encoding = face_recognition.face_encodings(image, boxes)[0]

doc = {
    "user": "Caio",
    "embedding": face_encoding,
}
res = elastic_con.index(index="users", id=1, body=doc)
