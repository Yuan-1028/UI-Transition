import os
import json
import csv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image
import random
import logging
import os
from logging import FileHandler, Formatter

def setup_logger(name, log_file, level=logging.INFO):
    """设置一个只写入文件的日志器，并在每次调用时删除旧的日志文件"""
    # 创建日志目录（如果不存在）
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 如果日志文件已存在，则删除它
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除所有已存在的处理器
    logger.handlers = []
    
    # 添加FileHandler
    handler = FileHandler(log_file)
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    # 禁用向上传播
    logger.propagate = False
    
    return logger

# 为每个主要函数创建独立的日志器
get_image_description_logger = setup_logger('get_image_description', 'logs/get_image_description.log')
validate_response_logger = setup_logger('validate_response', 'logs/validate_response.log')
calculate_accuracy_logger = setup_logger('calculate_accuracy', 'logs/calculate_accuracy.log')
target_description_logger = setup_logger('target_description', 'logs/target_description.log')

# Model settings
MODEL_PATH = "liuhaotian/llava-v1.6-vicuna-7b"

def setup_model():
    print("Setting up the model...")
    pipe = pipeline(MODEL_PATH, backend_config=TurbomindEngineConfig(tp=2, session_len=8192))
    return pipe

def get_image_description(image_path, pipe, gen_config, data_id):
    logger = get_image_description_logger
    query = 'Describe this image in detail, and the description should be between 15 to 80 words.'
    image = load_image(image_path)
    response = pipe((query, image), gen_config=gen_config)
    description = response.text.strip()
    logger.info(f"[Data ID: {data_id}] Generated description: {description}")
    return description

def parse_json_file(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        while color in colors:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def draw_bounding_boxes(image_path, json_data, output_path, font_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 生成足够多的不同颜色
    colors = generate_distinct_colors(len(json_data["compos"]))
    
    # 使用指定的字体，大小设为36
    font = ImageFont.truetype(font_path, 24)

    for idx, element in enumerate(json_data["compos"], 1):
        position = element
        
        column_min = position.get("column_min", 0)
        row_min = position.get("row_min", 0)
        column_max = position.get("column_max", 0)
        row_max = position.get("row_max", 0)
        
        box = (column_min, row_min, column_max, row_max)
        color = colors[idx - 1]  # 为每个框选择一个不同的颜色
        
        # 绘制边界框
        draw.rectangle(box, outline=color, width=2)
        
        # 绘制文本，位置在框的正上方
        text = str(idx)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (column_min + (column_max - column_min) // 2 - text_width // 2, 
                         max(0, row_min - text_height - 5))
        
        # 绘制文本
        draw.text(text_position, text, fill=color, font=font)

    image.save(output_path)
    return image

def get_model_response_for_target(start_image_path, target_description, pipe, gen_config, data_id):
    logger = target_description_logger
    prompt = f"What UI element should I tap to transit to the screen that contains {target_description}? Please tell me the index number of it."
    image = load_image(start_image_path)
    response = pipe((prompt, image), gen_config=gen_config)
    description = response.text.strip()
    logger.info(f"[Data ID: {data_id}] Generated description: {description}")
    return description

def validate_response(response, json_data, data_id):
    logger = validate_response_logger
    try:
        # 尝试从响应中提取数字
        index = int(''.join(filter(str.isdigit, response)))
        if 1 <= index <= len(json_data["compos"]):
            logger.info(f"[Data ID: {data_id}] Valid index found: {index}")
            return True, json_data["compos"][index - 1]
        else:
            logger.warning(f"[Data ID: {data_id}] Index out of range: {index}")
            return False, None
    except ValueError:
        logger.error(f"[Data ID: {data_id}] No valid index found in response")
        return False, None

def get_human_label(human_label_path):
    with open(human_label_path, 'r') as f:
        human_data = json.load(f)
    for element in human_data['compos']:
        if element.get('human_label', 0) == 1:
            return element
    return None


def calculate_accuracy(model_element, human_element, data_id):
    logger = calculate_accuracy_logger
    if not model_element or not human_element:
        logger.warning(f"[Data ID: {data_id}] Missing model_element or human_element")
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
    logger.info(f"[Data ID: {data_id}] IoU score: {iou_score}")
    return iou_score > 0.5

def process_pair(start_image, end_image, yolo_json, output_dir, pipe, description_config, target_config, data_id, font_path):
    target_screen_description = get_image_description(end_image, pipe, description_config, data_id)
    
    json_data = parse_json_file(yolo_json)
    
    annotated_image_path = os.path.join(output_dir, f"annotated_{os.path.basename(start_image)}")
    draw_bounding_boxes(start_image, json_data, annotated_image_path, font_path)
    
    model_response = get_model_response_for_target(start_image, target_screen_description, pipe, target_config, data_id)
    
    is_valid, model_element = validate_response(model_response, json_data, data_id)
    
    human_element = get_human_label(yolo_json)
    
    is_accurate = calculate_accuracy(model_element, human_element, data_id) if is_valid and human_element else False
    
    result = {
        "start_image": start_image,
        "end_image": end_image,
        "target_description": target_screen_description,
        "model_response": model_response,
        "is_valid": is_valid,
        "suggested_element": json.dumps(model_element) if model_element else None,
        "human_labeled_element": json.dumps(human_element) if human_element else None,
        "is_accurate": is_accurate,
        "annotated_image": annotated_image_path
    }
    
    return result

def process_dataset(dataset_dir, output_dir, pipe, description_config, target_config, font_path):
    results = []
    pairs = []
    total_pairs = 0
    accurate_pairs = 0

    for root, dirs, files in os.walk(dataset_dir):
        start_image = end_image = yolo_json = human_label_json = None
        for file in files:
            if file.endswith('start.jpg'):
                start_image = os.path.join(root, file)
            elif file.endswith('end.jpg'):
                end_image = os.path.join(root, file)
            elif file.startswith('YOLO') and file.endswith('start.json'):
                yolo_json = os.path.join(root, file)
        data_id = os.path.basename(root)
        if start_image and end_image and yolo_json and data_id:
            pairs.append((start_image, end_image, yolo_json, data_id))

    pbar = tqdm(pairs, desc="Processing image pairs")
    for idx, (start_image, end_image, yolo_json,data_id) in enumerate(pbar, 1):
        result = process_pair(start_image, end_image, yolo_json, output_dir, pipe, description_config, target_config, data_id, font_path)
        results.append(result)
        total_pairs += 1
        if result['is_accurate']:
            accurate_pairs += 1
        
        # 计算并显示运行中的准确率
        running_accuracy = accurate_pairs / total_pairs
        pbar.set_postfix(accuracy=f"{running_accuracy:.2%}", 
                         correct=f"{accurate_pairs}/{total_pairs}")
        logging.info(f"[Data ID: {data_id}] Running accuracy: {running_accuracy:.2%} ({accurate_pairs}/{total_pairs})")

    final_accuracy = accurate_pairs / total_pairs if total_pairs > 0 else 0
    logging.info(f"Final accuracy: {final_accuracy:.2%}")

    return results, final_accuracy

def save_results_to_csv(results, accuracy, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["start_image", "end_image", "target_description", "model_response", 
                      "is_valid", "suggested_element", "human_labeled_element", 
                      "is_accurate", "annotated_image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        
        writer.writerow({"start_image": "Overall Accuracy", "end_image": f"{accuracy:.2%}"})

def main():
    dataset_dir = "/root/task/UI-Transition/data/human_labeled"
    output_dir = "/root/task/UI-Transition/results"
    font_path = "Roboto-Bold.ttf"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    description_config = GenerationConfig(
        top_k=40,            # 保持不变，允许多样化的词汇选择
        top_p=0.9,           # 略微提高，以包含更多可能的相关描述
        temperature=0.7,     # 略微提高，增加创造性和多样性
        max_new_tokens=150,  # 允许生成更长的描述
        repetition_penalty=1.2,  # 增加重复惩罚，避免重复描述
        min_new_tokens=30    # 确保生成至少一定长度的描述
    )
    target_config = GenerationConfig(
        temperature=0.2,       # 降低温度以获得更确定的答案
        top_p=0.95,            # 保持较高的 top_p 以允许一定的灵活性
        top_k=5,               # 限制考虑的 token 数量，但保留一些选择
        repetition_penalty=1.1,# 轻微惩罚重复，确保不会重复输出数字
        min_new_tokens=1       # 确保至少生成一个 token（数字）
    )
    pipe = setup_model()
    
    results, accuracy = process_dataset(dataset_dir, output_dir, pipe, description_config, target_config, font_path)

    csv_output_file = os.path.join(output_dir, "analysis_results.csv")
    save_results_to_csv(results, accuracy, csv_output_file)

    print(f"Analysis complete. Results saved to {csv_output_file}")
    print(f"Overall accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()