"""
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/cache/

conda create -n llava python=3.10 -y
conda activate llava
"""

from lmdeploy import pipeline
import base64
import json
import os
import re
import pandas as pd

system_prompt = "You are an ordinary smartphone user who can understand the transition logic between consecutive GUI screens. You will be given a pair of consecutive smartphone GUI screens, you need to identify the index number of the UI element on the prior screen that link the prior UI screen to the next UI screen. If you fail to do this, also explain your reason."
system_knowledge_prompt = "You can reason the navigation relationship with the following 5 principles: \n Principle (1). Comparing semantic consistency, and choose the UI element that is related with the main topic of the next screen. \n Principle (2). Reasoning the logic or workflow and the hierarchical relationship beteen the two screens (e.g., go back to the home screen). \n Principle (3). Comparing the visual variant between two GUI screens and choose the UI element that got larger or highlighted. \n Principle (4). Understand and recognize the common navigation mode. \n Principle (5). Choosing the most visually salient UI element."

user_prompt = "Please describe how to transit from the prior UI screen to the later UI screen. You need to identify the index of the UI element that link the prior UI screen to the next UI screen, and explain your reason for such a choice. If you feel they are not consecutive screens or have no link UI, also explain your reason."


def encode_image(image_path):
    """
    Encodes an image file into a base64 string.
    Args:
        image_path (str): The path to the image file.

    This function opens the specified image file, reads its content, and encodes it into a base64 string.
    The base64 encoding is used to send images over HTTP as text.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
class LLavaModel:
    def __init__(self, dataset, model_name='liuhaotian/llava-v1.6-vicuna-7b'):
        self.model_name = model_name
        self.pipe = pipeline(model_name)
        self.dataset = dataset
    
    def get_summarization_from_hand_end_json(self, end_img_path):
        image_name = os.path.basename(end_img_path)
        # 正则匹配图片的序号
        image_num = re.findall(r"\d+", image_name)[0]
        # 读取json文件
        json_name = f"hand_{image_num}end.json"
        print(f"reading {json_name}")
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.dataset,
            json_name,
        )
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for item in json_data:
                if item.get("summarization", ""):
                    return item.get("summarization")
            return None
        
    def get_model_response_llava(self, prompt, start_end_images: list):
        target_des = self.get_summarization_from_hand_end_json(start_end_images[1])

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "text",
                "text": "Here is the description of the next UI screen.\n" + target_des,
            },
        ]
        img1 = encode_image(start_end_images[0])

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img1}"},
            }
        )

        messages = [
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        response = self.pipline(messages)
        return response
        
    def get_model_response_with_human_knowledge_llava(
        self, prompt, start_end_images: list
    ):
        target_des = self.get_summarization_from_hand_end_json(start_end_images[1])

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "text",
                "text": "Here is the description of the next UI screen.\n" + target_des,
            },
        ]
        img1 = encode_image(start_end_images[0])

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img1}"},
            }
        )

        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": system_knowledge_prompt},
            {"role": "user", "content": content},
        ]

        response = self.pipe(messages)

        return response.text
    
def promptLLava(mode, representation, dataset=""):
    mllm = LLavaModel(dataset=dataset)
    result_dict = {
        "pair_num": [],
        "result": [],
    }

    for i in range(1, 141):
        before = after = None
        if representation == "high_fidelity":
            before = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                dataset,
                str(i) + "mark_start.jpg",
            )
            after = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                dataset,
                str(i) + "mark_end.jpg",
            )
        elif representation == "wireframe":
            before = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                dataset,
                str(i) + "wireframe_mark_start.jpg",
            )
            after = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                dataset,
                str(i) + "wireframe_mark_end.jpg",
            )

        if os.path.exists(before) and os.path.exists(after):

            prompt = "Please describe how to transit from the given UI screen to the target UI screen described with natural language. You need to identify the index number of the UI element that link the prior UI screen to the next UI screen, and explain your reason for such a choice. If you feel they are not consecutive screens or have no link UI, also explain your reason."

            rsp = None
            if mode == "direct_prompt":
                rsp = mllm.get_model_response_llava(prompt, [before, after])
            if mode == "prompt_with_knowledge":
                rsp = mllm.get_model_response_with_human_knowledge_llava(
                    prompt, [before, after]
                )

            print(i, rsp)
            result_dict["pair_num"].append(i)
            result_dict["result"].append(rsp)

            df = pd.DataFrame(result_dict)
            df.to_csv(mode + "_" + dataset + "llava.csv", index=False)

if __name__ == "__main__":
    modes = ["direct_prompt", "prompt_with_knowledge"]
    datasets = ["test_finetune"]
    representation = ["high_fidelity", "wireframe"]
    promptLLava(modes[1], representation[0], datasets[0])
