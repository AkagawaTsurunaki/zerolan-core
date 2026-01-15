"""
More details about Milvus:
    https://milvus.io/docs
"""
from flask import Flask, jsonify, request
from typing import Type

from milvus_model import DefaultEmbeddingFunction
from pymilvus import MilvusClient
from pydantic import BaseModel
from loguru import logger
from zerolan.data.pipeline.milvus import MilvusInsert, MilvusInsertResult, MilvusQuery, QueryRow, MilvusQueryResult

from database.milvus.config import MilvusDBConfig


class MilvusDatabase:
    def __init__(self, config: MilvusDBConfig):
        self._milvus_path: str = config.db_path
        self._dimension = 768
        self._client: MilvusClient = MilvusClient(self._milvus_path)
        self._embedding_fn = DefaultEmbeddingFunction()

    def try_create_collection(self, collection_name: str, dimension: int, overwrite: bool = False):
        if self._client.has_collection(collection_name=collection_name):
            if overwrite:
                self._client.drop_collection(collection_name=collection_name)

        if not self._client.has_collection(collection_name=collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                auto_id=True
            )

    def insert(self, insert: MilvusInsert) -> MilvusInsertResult:
        assert isinstance(insert, MilvusInsert)
        self.try_create_collection(
            insert.collection_name, self._dimension, insert.drop_if_exists)
        docs = [row.text for row in insert.texts]
        vectors = self._embedding_fn.encode_documents(docs)
        data = [
            {"id": i, "vector": vectors[i],
                "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        res = self._client.insert(
            collection_name=insert.collection_name, data=data)
        logger.info(res)
        return MilvusInsertResult(insert_count=res.get(
            "insert_count", -1), ids=res.get("ids", []))

    def search(self, query: MilvusQuery):
        assert isinstance(query, MilvusQuery)
        query_vectors = self._embedding_fn.encode_queries([query.query])
        data = self._client.search(
            collection_name=query.collection_name,  # target collection
            data=query_vectors,  # query vectors
            limit=query.limit,  # number of returned entities
            output_fields=query.output_fields  # specifies fields to be returned
        )

        result = []
        for record in data:
            rows = []
            for item in record:
                row = QueryRow(id=item["id"], distance=item["distance"],
                                 entity=item["entity"])
                rows.append(row)
            result.append(rows)

        return MilvusQueryResult(result=result)


class MilvusApplication:
    def __init__(self, database: MilvusDatabase, host: str, port: int):
        self.host = host
        self.port = port
        self._app = Flask(__name__)
        self._app.add_url_rule(rule="/milvus/insert",
                               view_func=self._handle_insert, methods=["POST"])
        self._app.add_url_rule(rule="/milvus/search",
                               view_func=self._handle_search, methods=["POST"])

        self._database = database

    def _from_json(self, t: Type[BaseModel]):
        with self._app.app_context():
            json_val = request.get_json()
            obj = t.model_validate(json_val)
            return obj

    def _to_json(self, res: any):
        if isinstance(res, BaseModel):
            res = res.model_dump()

        return jsonify(res)

    def _handle_insert(self):
        insert: MilvusInsert = self._from_json(MilvusInsert)
        res = self._database.insert(insert)
        return self._to_json(res)

    def _handle_search(self):
        query: MilvusQuery = self._from_json(MilvusQuery)
        res = self._database.search(query)
        return self._to_json(res)

    def run(self):
        self._app.run(self.host, self.port, False)
