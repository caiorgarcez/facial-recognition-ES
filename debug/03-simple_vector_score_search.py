import argparse

import cv2
import face_recognition
import numpy as np
from elasticsearch import Elasticsearch
from scipy.spatial import distance as dist

es = Elasticsearch([{"host": "localhost"}])

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


def get_username(id, index):
    """retreive an user from users index on ES according to a given id

    Args:
        id (integer): desired user's id.
        index (string): users index on ES.
    """
    res = es.get(index=index, id=id)
    return res["_source"]


def build_vector_query(query_vec, vector_field):
    score_function = (
        "doc['{v}'].size() == 0 ? 0 : 1 / (1 + l2norm(params.queryVector, '{v}'))"
    )

    score_function = score_function.format(v=vector_field, fn=score_function)

    return {
        "query": {
            "script_score": {
                "query": {"query_string": {"query": "*"}},
                "script": {
                    "source": score_function,
                    "params": {"queryVector": query_vec},
                },
            }
        }
    }


def main():

    if args["source"]:
        face_encoding = get_face_encoding(args["source"])

    embedding_vector = get_username(1, "users").get("embedding")

    matches = face_recognition.compare_faces(
        [face_encoding], np.array(embedding_vector)
    )

    query = build_vector_query(query_vec=face_encoding, vector_field="embedding")

    results = es.search(index="users", body=query)

    print(results)


if __name__ == "__main__":
    main()
