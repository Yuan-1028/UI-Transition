import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm.asyncio import tqdm
import random
from openai import AsyncOpenAI
import aiofiles
import base64
from aiocsv import AsyncDictWriter
from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.files import AsyncFileHandler
import asyncio

async def setup_logger(name, log_file, level="INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    
    logger = Logger(name=name, level=level)
    
    handler = AsyncFileHandler(filename=log_file)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.formatter = formatter
    
    logger.add_handler(handler)
    
    return logger

# 使用异步函数来设置日志器
async def setup_all_loggers():
    global get_image_description_logger, validate_response_logger, calculate_accuracy_logger, target_description_logger
    
    get_image_description_logger = await setup_logger('get_image_description', 'logs/get_image_description.log')
    validate_response_logger = await setup_logger('validate_response', 'logs/validate_response.log')
    calculate_accuracy_logger = await setup_logger('calculate_accuracy', 'logs/calculate_accuracy.log')
    target_description_logger = await setup_logger('target_description', 'logs/target_description.log')

# 关闭所有日志器的函数
async def close_all_loggers():
    await asyncio.gather(
        get_image_description_logger.shutdown(),
        validate_response_logger.shutdown(),
        calculate_accuracy_logger.shutdown(),
        target_description_logger.shutdown()
    )

async def encode_image(image_path):
    async with aiofiles.open(image_path, "rb") as image_file:
        image_data = await image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

async def get_image_description(client, image_path, data_id):
    logger = get_image_description_logger
    base64_image = await encode_image(image_path)
    response = await client.chat.completions.create(
        model="cogvlm2",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the image in detail, using between 15 to 30 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
        max_tokens=30
    )
    description = response.choices[0].message.content.strip()
    await logger.info(f"[Data ID: {data_id}] Generated description: {description}")
    return description

async def parse_json_file(json_path):
    async with aiofiles.open(json_path, "r") as json_file:
        json_content = await json_file.read()
    return json.loads(json_content)

async def get_model_response_for_target(client, start_image_path, target_description, data_id):
    logger = target_description_logger
    base64_image = await encode_image(start_image_path)
    response = await client.chat.completions.create(
        model="cogvlm2",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'''What UI element should I tap to transit to the screen that contains {target_description}? 
Please tell me the index number of it.'''
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
        max_tokens=150
    )
    description = response.choices[0].message.content.strip()
    await logger.info(f"[Data ID: {data_id}] Generated description: {description}")
    return description

async def validate_response(response, json_data, data_id):
    logger = validate_response_logger
    try:
        index = int(''.join(filter(str.isdigit, response)))
        if 1 <= index <= len(json_data["compos"]):
            await logger.info(f"[Data ID: {data_id}] Valid index found: {index}")
            return index - 1, json_data["compos"][index - 1]
        else:
            await logger.warning(f"[Data ID: {data_id}] Index out of range: {index}")
            return 0, None
    except ValueError:
        await logger.error(f"[Data ID: {data_id}] No valid index found in response")
        return 0, None

def get_human_label(human_data):
    for index,element in enumerate(human_data['compos']):
        if element.get('human_label', 0) == 1:
            return index, element
    return 0, None

async def calculate_accuracy(model_element, human_element, data_id):
    logger = calculate_accuracy_logger
    if not model_element or not human_element:
        await logger.warning(f"[Data ID: {data_id}] Missing model_element or human_element")
        return False
    
    def get_box(element):
        return (
            element.get("column_min", 0),
            element.get("row_min", 0),
            element.get("column_max", 0),
            element.get("row_max", 0)
        )

    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        intersect_x1 = max(x1, x3)
        intersect_y1 = max(y1, y3)
        intersect_x2 = min(x2, x4)
        intersect_y2 = min(y2, y4)
        
        intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersect_area
        
        if union_area == 0:
            return 0
        
        return intersect_area / union_area

    box1 = get_box(model_element)
    box2 = get_box(human_element)
    iou_score = iou(box1, box2)
    await logger.info(f"[Data ID: {data_id}] IoU score: {iou_score}")
    return iou_score > 0.5

async def save_result_to_csv(result, writer):
    await writer.writerow(result)

async def process_pair(client, start_image, end_image, yolo_json, data_id, writer):
    target_screen_description = await get_image_description(client, end_image, data_id)
    
    json_data = await parse_json_file(yolo_json)
    
    model_response = await get_model_response_for_target(client, start_image, target_screen_description, data_id)
    
    index, model_element = await validate_response(model_response, json_data, data_id)
    
    human_index, human_element = get_human_label(json_data)
    
    is_accurate = await calculate_accuracy(model_element, human_element, data_id) if index > 0 and human_element else False
    
    result = {
        "start_image": start_image,
        "end_image": end_image,
        "target_description": target_screen_description,
        "model_response": model_response,
        "suggested_element": json.dumps(model_element) if model_element else None,
        "human_labeled_element": json.dumps(human_element) if human_element else None,
        "is_accurate": is_accurate
    }
    
    await save_result_to_csv(result, writer)
    
    return result

async def process_dataset(client, dataset_dir, sample_limit=None, csv_output_file="logs/results.csv"):
    pairs = []
    total_pairs = 0
    accurate_pairs = 0
    
    for root, _, files in os.walk(dataset_dir):
        start_image = end_image = yolo_json = None
        for file in files:
            if file.endswith('start.jpg'):
                start_image = os.path.join(root, file)
            elif file.endswith('end.jpg'):
                end_image = os.path.join(root, file)
            elif file.endswith('.json'):
                yolo_json = os.path.join(root, file)
        data_id = os.path.basename(root)
        if start_image and end_image and yolo_json and data_id:
            pairs.append((start_image, end_image, yolo_json, data_id))

    if sample_limit > 0 and sample_limit < len(pairs):
        pairs = random.sample(pairs, sample_limit)

    fieldnames = ["start_image", "end_image", "target_description", "model_response", 
                  "suggested_element", "human_labeled_element", 
                  "is_accurate", "annotated_image"]

    async with aiofiles.open(csv_output_file, mode='w+', newline='', encoding='utf-8') as csvfile:
        writer = AsyncDictWriter(csvfile, fieldnames=fieldnames)
        await writer.writeheader()

        async def process_pair_wrapper(pair):
            nonlocal total_pairs, accurate_pairs
            start_image, end_image, yolo_json, data_id = pair
            result = await process_pair(client, start_image, end_image, yolo_json, data_id, writer)
            total_pairs += 1
            if result['is_accurate']:
                accurate_pairs += 1
            running_accuracy = accurate_pairs / total_pairs
            return result

        results = []
        for result in tqdm.as_completed([process_pair_wrapper(pair) for pair in pairs],
                                              total=len(pairs),
                                              desc="Processing image pairs"):
            results.append(await result)
            running_accuracy = accurate_pairs / total_pairs
            tqdm.write(f"Current accuracy: {running_accuracy:.2%} ({accurate_pairs}/{total_pairs})")

        final_accuracy = accurate_pairs / total_pairs if total_pairs > 0 else 0
        await writer.writerow({"start_image": "Overall Accuracy", "end_image": f"{final_accuracy:.2%}"})

    return results, final_accuracy

async def main():
    dataset_dir = "process_data"
    output_dir = "logs"
    sample_limit = 0
    
    await setup_all_loggers()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = AsyncOpenAI(
        api_key="sb-99b4589db7cadab4289a42a142291172aff46c79fb8c927b",
        base_url="http://127.0.0.1:8080/v1/"
    )

    csv_output_file = os.path.join(output_dir, "analysis_results.csv")
    results, accuracy = await process_dataset(client, dataset_dir, sample_limit, csv_output_file)

    print(f"Analysis complete. Results saved to {csv_output_file}")
    print(f"Overall accuracy: {accuracy:.2%}")
    
    await close_all_loggers()

if __name__ == "__main__":
    asyncio.run(main())