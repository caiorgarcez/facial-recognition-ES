import argparse
import os
import pickle

import cv2
import face_recognition

ap = argparse.ArgumentParser()
ap.add_argument(
    "-isrc", "--source", required=False, help="Full path for image source file."
)
args = vars(ap.parse_args())


def get_face_encoding(image_path):
    """Generate a face_encoding from a given image path

    Args:
        image_path (str): image file path

    Returns:
        np.array: face embedding array
    """
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    return face_recognition.face_encodings(image, boxes)[0]


if args["source"]:
    face_encoding = get_face_encoding(args["source"])
    with open("output.pkl", "wb") as f:
        pickle.dump(face_encoding, f)

print(f"[INFO] Embedding vector:\n {face_encoding}\n")
print(f"[INFO] Length: {len(face_encoding)}")
