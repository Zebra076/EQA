import time
import json
import base64
import os

import cv2
import torch
from openai import OpenAI


class LLMRequest:

    def __init__(self, config_path='config.json', api_type=None):

        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.base_url = config["base_url"]
        self.api_key = config["api_key"]
        self.model = config["model"]

        self.api_type = self._normalize_api_type(
            api_type or config.get("api_type", "completions")
        )

        print(f"Using model: {self.model} with API type: {api_type or config.get('api_type', 'completions')}")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def _normalize_api_type(self, api_type):
        api_type = api_type.lower().replace("-", "_")
        if api_type in [
            "completion", "completions", "chat_completion",
            "chat_completions", "chat.completions",
        ]:
            return "completions"
        if api_type in ["response", "responses"]:
            return "responses"
        raise ValueError("api_type must be 'completions' or 'responses'")

    @staticmethod
    def read_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def rgb_image_to_base64_png(img_rgb):
        if img_rgb is None:
            raise ValueError("img_rgb must not be None")

        if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
            img_to_encode = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_to_encode = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGRA)
        else:
            img_to_encode = img_rgb

        ok, encoded = cv2.imencode(".png", img_to_encode)
        if not ok:
            raise RuntimeError("failed to encode observation image")
        return base64.b64encode(encoded).decode("utf-8")

    def _normalize_base64_images(self, base64_images):
        if not isinstance(base64_images, dict):
            if isinstance(base64_images, list):
                base64_images = {(i+1):image for i, image in enumerate(base64_images)}
            elif isinstance(base64_images, str):
                base64_images = {1: base64_images}
            else:
                raise ValueError("base64_images must be a str, a list or dict")
        return base64_images

    def _build_completions_messages(self, prompt, base64_images):
        content = []

        content.append({
            'type': 'text',
            'text': prompt,
        })

        for i, image in base64_images.items():
            content.append({
                'type': 'text',
                'text': f'Picture {i}:',
            })
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': self._to_image_url(image),
                },
            })

        return [{
            'role': 'user',
            'content': content,
        }]

    def _build_responses_input(self, prompt, base64_images):
        content = []

        content.append({
            'type': 'input_text',
            'text': prompt,
        })

        for i, image in base64_images.items():
            content.append({
                'type': 'input_text',
                'text': f'Picture {i}:',
            })
            content.append({
                'type': 'input_image',
                'image_url': self._to_image_url(image),
                'detail': 'auto',
            })

        return [{
            'role': 'user',
            'content': content,
        }]

    def _detect_image_mime(self, image):
        """根据 base64 解码后的文件头(magic number)判断图片格式，返回 mime 类型。"""
        try:
            # 只需解码前若干字节即可判断，截取一小段避免解码整张大图
            header = base64.b64decode(image[:32], validate=False)
        except Exception:
            # 解码失败时退回到 png
            return "image/png"

        # 各格式的文件头特征
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        if header.startswith(b"BM"):
            return "image/bmp"

        # 未知格式默认 png
        return "image/png"

    def _to_image_url(self, image):
        if image.startswith(("data:", "http://", "https://")):
            return image
        mime = self._detect_image_mime(image)
        return f"data:{mime};base64,{image}"

    def _extract_response_text(self, response):
        if hasattr(response, "output_text"):
            return response.output_text

        texts = []
        for output in getattr(response, "output", []) or []:
            for content in getattr(output, "content", []) or []:
                text = getattr(content, "text", None)
                if text is not None:
                    texts.append(text)
        return "".join(texts)

    def generate_response(self, prompt, base64_images={}, **kwargs):
        base64_images = self._normalize_base64_images(base64_images)

        if self.api_type == "responses":
            response = self.client.responses.create(
                model=self.model,
                input=self._build_responses_input(prompt, base64_images),
                stream=False,
                **kwargs
            )
            return self._extract_response_text(response)

        messages = self._build_completions_messages(prompt, base64_images)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content
        

if __name__ == "__main__":
    import airsim

    try:
        from .airsim_interface import AirSimInterface
    except ImportError:
        from airsim_interface import AirSimInterface


    current_dir = os.path.dirname(os.path.abspath(__file__))
    obs_save_path = os.path.join(current_dir, "vlm_obs_test.png")

    airsim_interface = AirSimInterface(img_save_path=obs_save_path, client_port=41453)
    poses = [
        airsim.Pose(
            airsim.Vector3r(0, 0, -2),
            airsim.to_quaternion(0, 0, 0),
        )
    ]
    imgs_rgb = airsim_interface.get_obs_imgs(poses, save=obs_save_path)
    if not imgs_rgb:
        raise RuntimeError("get_obs_imgs did not return any images")

    base64_images = [LLMRequest.rgb_image_to_base64_png(img_rgb) for img_rgb in imgs_rgb]

    # base64_images = [LLMRequest.read_image_to_base64("src/common/tmp/get_obs_img.png") for img_rgb in imgs_rgb]

    llm = LLMRequest()
    response = llm.generate_response(
        "Describe the image.",
        base64_images,
    )
    print(response)
