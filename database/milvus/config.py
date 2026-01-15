from dataclasses import dataclass


@dataclass
class MilvusDBConfig:
    db_path: str = "./.data/milvus.db"
    host: str = "127.0.0.1"
    port: int = 11010
