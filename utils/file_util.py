import os
import uuid
from time import time
from typing import Literal

project_dir = os.getcwd()
temp_data_dir = os.path.join(project_dir, "/.temp")
if not os.path.exists(temp_data_dir):
    os.makedirs(temp_data_dir)


def create_temp_file(prefix: str, suffix: str, tmpdir: Literal["image", "video", "audio"]) -> str:
    tmp_dir = os.path.join(temp_data_dir, tmpdir)
    tmp_dir = os.path.abspath(tmp_dir)
    temp_file_path = os.path.join(f"{tmp_dir}", f"{prefix}-{time()}-{uuid.uuid4()}{suffix}")
    temp_file_path = temp_file_path.replace("\\", "/")
    temp_file_path = os.path.abspath(temp_file_path)
    return temp_file_path
