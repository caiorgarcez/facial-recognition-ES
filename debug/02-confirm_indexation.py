from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost"}])


def get_username(id, index):
    """retreive an user from users index on ES according to a given id

    Args:
        id (integer): desired user's id.
        index (string): users index on ES.
    """
    res = es.get(index=index, id=id)
    print(res["_source"])


def main():
    get_username(1, "users")


if __name__ == "__main__":
    main()
