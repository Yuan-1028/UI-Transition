import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
import logging
from logging import FileHandler, Formatter
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image

def setup_logger(name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    handler = FileHandler(log_file)
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger

dataset_logger = setup_logger('dataset_builder', 'logs/dataset_builder.log')

class ConversationDatasetBuilder:
    def __init__(self, source_dir, output_dir, model_path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        self.logger = dataset_logger
        self.pipe = self.setup_model(model_path)

    def setup_model(self, model_path):
        print("Setting up the CogVLM2 model...")
        return pipeline(model_path, backend_config=TurbomindEngineConfig(tp=2, session_len=8192))

    def setup_output_dirs(self):
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def parse_json_file(self, json_path):
        with open(json_path, "r") as json_file:
            return json.load(json_file)

    def get_human_label_and_index(self, json_data):
        for index, element in enumerate(json_data['compos'], 1):
            if element.get('human_label', 0) == 1:
                return element, index
        return None, None

    def get_image_description(self, image_path, gen_config):
        query = 'Describe this image in detail, and the description should be between 15 to 80 words.'
        image = load_image(image_path)
        response = self.pipe((query, image), gen_config=gen_config)
        return response.text.strip()

    def process_image_pair(self, start_image, end_image, json_path, data_id, description_config):
        try:
            json_data = self.parse_json_file(json_path)
            human_labeled_element, element_index = self.get_human_label_and_index(json_data)

            if not human_labeled_element:
                self.logger.warning(f"No human-labeled element found for {data_id}")
                return None, None, None, None

            # 使用 CogVLM2 生成 end.jpg 的描述
            end_screen_description = self.get_image_description(end_image, description_config)

            # 创建图片总结对话任务（使用 end.jpg）
            summary_task = {
                "task_type": "summary",
                "conversations": [
                    {
                        "role": "user",
                        "content": "Please provide a brief summary of this UI screen, focusing on its main purpose and key elements."
                    },
                    {
                        "role": "assistant",
                        "content": end_screen_description
                    }
                ]
            }

            # 创建识别跳转区域的对话任务
            identification_task = {
                "task_type": "identification",
                "conversations": [
                    {
                        "role": "user",
                        "content": f"The next screen shows: {end_screen_description}. What UI element should I tap to navigate to this screen? Please provide the index number of the element."
                    },
                    {
                        "role": "assistant",
                        "content": f"Please tap the element with index {element_index}."
                    }
                ]
            }
            
            return summary_task, identification_task, end_image, start_image
        except Exception as e:
            self.logger.error(f"Error processing image pair {data_id}: {str(e)}")
            return None, None, None, None

    def build_dataset(self, description_config):
        self.setup_output_dirs()
        sample_count = 0

        for root in tqdm(os.listdir(self.source_dir), desc="Processing data"):
            root_path = os.path.join(self.source_dir, root)
            if not os.path.isdir(root_path):
                continue
            
            files = os.listdir(root_path)
            start_image = end_image = json_file = None
            for file in files:
                if file.endswith('start.jpg'):
                    start_image = os.path.join(root_path, file)
                elif file.endswith('end.jpg'):
                    end_image = os.path.join(root_path, file)
                elif file.endswith('.json'):
                    json_file = os.path.join(root_path, file)
            
            if start_image and end_image and json_file:
                summary_task, identification_task, end_img, start_img = self.process_image_pair(start_image, end_image, json_file, root, description_config)
                if summary_task and identification_task and end_img and start_img:
                    # 为结束图像（总结任务）生成八位数字文件名
                    end_image_name = f"{(sample_count * 2 + 1):08d}.jpg"
                    end_label_name = f"{(sample_count * 2 + 1):08d}.json"
                    shutil.copy(end_img, os.path.join(self.images_dir, end_image_name))
                    with open(os.path.join(self.labels_dir, end_label_name), 'w') as f:
                        json.dump(summary_task, f, indent=2)
                    
                    # 为起始图像（识别任务）生成八位数字文件名
                    start_image_name = f"{(sample_count * 2 + 2):08d}.jpg"
                    start_label_name = f"{(sample_count * 2 + 2):08d}.json"
                    shutil.copy(start_img, os.path.join(self.images_dir, start_image_name))
                    with open(os.path.join(self.labels_dir, start_label_name), 'w') as f:
                        json.dump(identification_task, f, indent=2)
                    
                    sample_count += 1
                    self.logger.info(f"Processed and saved sample pair: {end_image_name} and {start_image_name}")

        self.logger.info(f"Total sample pairs in dataset: {sample_count}")
        return sample_count * 2  # 返回总样本数（每对图像生成两个样本）

def main():
    source_dir = "process_data"
    output_dir = "data/sft_data"
    model_path = "THUDM/cogvlm2-llama3-chat-19B"
    
    description_config = GenerationConfig(
        top_k=40, top_p=0.9, temperature=0.7, max_new_tokens=150,
        repetition_penalty=1.2, min_new_tokens=30
    )
    
    builder = ConversationDatasetBuilder(source_dir, output_dir, model_path)
    total_samples = builder.build_dataset(description_config)

    print(f"Dataset creation complete. Total samples: {total_samples}")
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    main()