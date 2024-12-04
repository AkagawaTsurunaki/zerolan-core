"""
More details about the model:
    https://github.com/showlab/ShowUI
    https://huggingface.co/showlab/ShowUI-2B
"""

# DeepSpeed lib is only supported on Linux platform!

from common.decorator import log_model_loading
from common.abs_model import AbstractModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import ast
from zerolan.data.pipeline.vla import ShowUiQuery, ShowUiPrediction, WebAction, PhoneAction
import os

from vla.showui.config import ShowUIModelConfig


_SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."


_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""

_NAV_FORMAT = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

action_map = {
    'web': """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
""",

    'phone': """
1. `INPUT`: Type a string into an element, value is not applicable and the position [x,y] is required. 
2. `SWIPE`: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
3. `TAP`: Tap on an element, value is not applicable and the position [x,y] is required.
4. `ANSWER`: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
5. `ENTER`: Enter operation, value and position are not applicable.
"""
}


class ShowUIModel(AbstractModel):
    def __init__(self, config: ShowUIModelConfig):
        self._model = None
        self._processor = None

        assert os.path.exists(
            config.showui_model_path), f"ShowUI model is not found at: {config.showui_model_path}"
        assert os.path.exists(
            config.qwen_vl_2b_instruct_model_path), f"Qwen VL 2B Instruct model is not found at: {config.qwen_vl_2b_instruct_model_path}"
        self._showui_model_path = config.showui_model_path
        self._showui_model_device = config.showui_model_device
        self._qwen_vl_2b_instruct_model_path = config.qwen_vl_2b_instruct_model_path
        self._min_pixels: int = config.min_pixels
        self._max_pixels: int = config.max_pixels

    @log_model_loading("showlab/ShowUI-2B")
    def load_model(self):
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._showui_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self._showui_model_device
        )
        self._processor = AutoProcessor.from_pretrained(
            self._qwen_vl_2b_instruct_model_path, min_pixels=self._min_pixels, max_pixels=self._max_pixels)

    def _predict_nav(self, query: ShowUiQuery) -> ShowUiPrediction:
        img_url = query.img_path
        split = query.action.env
        if query.system_prompt is None:
            system_prompt = _NAV_SYSTEM.format(
                _APP=split, _ACTION_SPACE=action_map[split])
        else:
            system_prompt = query.system_prompt
        query_content = query.query

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": f'Task: {query_content}'},
                    # {"type": "text", "text": PAST_ACTION},
                    {"type": "image", "image": img_url,
                        "min_pixels": self._min_pixels, "max_pixels": self._max_pixels},
                ],
            }
        ]

        output_text = self._predict(messages)
        # {'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]},
        # {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]},
        # {'action': 'ENTER', 'value': None, 'position': None}
        print(type(output_text))
        output_text = "[" + output_text + "]"

        # 使用ast.literal_eval()函数将字符串转换为Python的字典列表
        dicts = ast.literal_eval(output_text)
        print(type(dicts))
        print(output_text)

        assert isinstance(dicts, list)
        actions = []
        for elm in dicts:
            assert isinstance(elm, dict)
            action = elm.get('action')
            value = elm.get('value')
            position = elm.get('position')
            if split == 'web':
                actions.append(
                    WebAction(action=action, value=value, position=position))
            elif split == 'phone':
                actions.append(PhoneAction(
                    action=action, value=value, position=position))
            else:
                raise ValueError("No such platform!")

        return ShowUiPrediction(actions=actions)

    def _predict_ground(self, query: ShowUiQuery) -> ShowUiPrediction:
        img_url = query.img_path
        query_content = query.query
        if query.system_prompt is None:
            system_prompt = _SYSTEM
        else:
            system_prompt = query.system_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image", "image": img_url,
                        "min_pixels": self._min_pixels, "max_pixels": self._max_pixels},
                    {"type": "text", "text": query_content}
                ],
            }
        ]

        output_text = self._predict(messages)

        click_xy: list[int] = ast.literal_eval(output_text)
        # [0.73, 0.21]
        assert isinstance(click_xy, list)
        assert len(click_xy) == 2
        for elm in click_xy:
            assert isinstance(elm, float)

        action = WebAction(action="CLICK", value=None, position=click_xy)
        return ShowUiPrediction(actions=[action])

    def _predict(self, messages) -> str:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self._model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def predict(self, query: ShowUiQuery):
        if query.action is None:
            return self._predict_ground(query)
        else:
            return self._predict_nav(query)

    def stream_predict(self, *args, **kwargs):
        raise NotImplementedError()
