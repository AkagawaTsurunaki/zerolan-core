from dataclasses import dataclass


@dataclass
class MilvusDBConfig:
    db_path: str = "./.data/milvus.db"
