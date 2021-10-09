import argparse

import numpy as np
from elasticsearch import Elasticsearch

elastic_con = Elasticsearch([{"host": "localhost"}])

ap = argparse.ArgumentParser()
ap.add_argument(
    "-f", "--file", required=True, help="Full path for pickle embedings file."
)
args = vars(ap.parse_args())


def create_ES_index():
    """create an index on Elasticsearch according to a predefined mapping

    Returns:
        response from ES
    """
    # create index mapping for 128 n-dims embedding
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {"type": "dense_vector", "dims": 128},
                "user": {"type": "keyword"},
            }
        }
    }
    return elastic_con.indices.create(index="users", ignore=400, body=mapping)


def post_user_to_ES_index(username, embedding, user_id):
    """Create an user in the user's index on Elasticsearch

    Args:
        username (string): User's name.
        embedding (array): User's 128 n-dims array.
        user_id [integer]: Desired id number for the provided user.

    Returns:
        response from ES
    """
    default_doc = {
        "user": username,
        "embedding": embedding,
    }

    return elastic_con.index(index="users", id=int(user_id), body=default_doc)


def main():
    index_creation_output = create_ES_index()
    print(f"[INFO] {index_creation_output}")

    face_encoding = np.load(args["file"], allow_pickle=True)
    response_ES = post_user_to_ES_index("Caio", face_encoding, 1)
    print(f"[INFO] {response_ES}")


if __name__ == "__main__":
    main()
