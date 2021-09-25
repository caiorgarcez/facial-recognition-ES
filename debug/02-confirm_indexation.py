from elasticsearch import Elasticsearch

es = Elasticsearch([{"host": "localhost"}])

res = es.get(index="users", id=1)
print(res["_source"])


# REFERENCE: https://www.elastic.co/guide/en/elasticsearch/reference/7.x/query-dsl-script-score-query.html
