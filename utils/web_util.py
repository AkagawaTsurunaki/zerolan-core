from flask import Request
from pydantic import BaseModel

from utils.file_util import create_temp_file
from loguru import logger

def get_obj_from_json(request: Request, type: BaseModel) -> any:
    # If it's in multipart/form-data format, then try to get the deserialized JSON object
    json_str = request.form.get("json", None)
    if json_str is None:
        raise ValueError('There is no JSON data in the json field in the request')
    obj = type.model_validate_json(json_str)
    return obj


def save_request_image(request: Request, prefix: str) -> str:
    image_file = request.files.get('image', None)
    if image_file is None:
        raise ValueError('There is no image data in the image field in the request')

    file_type = image_file.mimetype.split("/")[-1]
    if len(file_type) != 0:
        assert file_type in ["jpg", "png"], "Unsupported image types"
    else:
        file_type = "png"

    temp_file_path = create_temp_file(prefix=prefix, suffix=f".{file_type}", tmpdir="image")
    image_file.save(temp_file_path)
    logger.debug(f"Temporary files are created at:{temp_file_path}")

    return temp_file_path


def get_request_audio_file(request: Request):
    # If it's in multipart/form-data format, then try to get the audio file
    audio_file = request.files.get('audio', None)
    if audio_file is None:
        raise ValueError('There is no audio data in the audio field in the request')
    return audio_file


def save_request_audio(request: Request, prefix: str) -> str:
    # If it's in multipart/form-data format, then try to get the audio file
    audio_file = get_request_audio_file(request)

    file_type = audio_file.mimetype.split("/")[-1]
    if len(file_type) != 0:
        if file_type == "wave":
            file_type = "wav"
        assert file_type in ["wav", "mp3", "ogg"], "Unsupported audio type"
    else:
        file_type = "wav"

    temp_file_path = create_temp_file(prefix=prefix, suffix=f".{file_type}", tmpdir="audio")
    audio_file.save(temp_file_path)
    logger.debug(f"Temporary files are created at: {temp_file_path}")

    return temp_file_path
