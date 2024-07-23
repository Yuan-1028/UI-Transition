
import os
import time
import torch
import jsonimport os
import time
import torch
import json
import csv
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 模型设置
MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_model():
    print("Setting up the model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 使用 Accelerate 库来加载大型模型
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model = load_checkpoint_and_dispatch(
        model,
        MODEL_PATH,
        device_map="auto",
        no_split_module_classes=["CogVLMModel"],
        dtype=torch.float16
    )

    model.eval()
    return tokenizer, model

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        return item.to(tgt)
    elif isinstance(item, (list, tuple)):
        return type(item)([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer):
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)

    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature

    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }

    if images:
        batch['images'] = images

    return batch

def get_image_description(image_path, model, tokenizer):
    query = 'Describe this image in detail, and the description should be between 15 to 80 words.'
    image = Image.open(image_path).convert('RGB')
    input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    input_batch = collate_fn([input_sample], tokenizer)

    # 确保所有输入都在正确的设备上
    input_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in input_batch.items()}

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": 1,
    }

    try:
        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(outputs)
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        print("GPU Memory Usage:")
        print(torch.cuda.memory_summary(device=device))
        return "Error: Unable to generate description."

    return outputs[0].split("<|end_of_text|>")[0].strip()

def parse_json_file(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)

def draw_bounding_boxes(image_path, json_data, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for idx, element in enumerate(json_data["compos"], 1):
        box = element["bounding_box"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(idx), fill="red")

    image.save(output_path)
    return image

def generate_text_prompt(target_description):
    return f"What UI element should I tap to transit to the screen that contains {target_description}? Please tell me the index number of it."

def get_model_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def validate_response(response, json_data):
    try:
        index = int(response) - 1
        if 0 <= index < len(json_data["compos"]):
            return True, json_data["compos"][index]
        else:
            return False, None
    except ValueError:
        return False, None

def get_human_label(human_label_path):
    with open(human_label_path, 'r') as f:
        human_data = json.load(f)
    for element in human_data['compos']:
        if element.get('is_correct', False):
            return element
    return None

def calculate_accuracy(model_element, human_element):
    if not model_element or not human_element:
        return False

    def iou(box1, box2):
        # 计算两个边界框的交并比（IoU）
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 计算交集区域
        intersect_x1 = max(x1, x3)
        intersect_y1 = max(y1, y3)
        intersect_x2 = min(x2, x4)
        intersect_y2 = min(y2, y4)

        intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

        # 计算并集区域
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersect_area

        if union_area == 0:
            return 0

        return intersect_area / union_area

    # 如果IoU大于0.5，我们认为预测是准确的
    return iou(model_element['bounding_box'], human_element['bounding_box']) > 0.5

def process_pair(start_image, end_image, yolo_json, human_label_json, output_dir, model, tokenizer):
    # 1. 获取目标屏幕描述
    target_screen_description = get_image_description(end_image, model, tokenizer)

    # 2. 解析JSON文件
    json_data = parse_json_file(yolo_json)

    # 3. 绘制边界框
    annotated_image_path = os.path.join(output_dir, f"annotated_{os.path.basename(start_image)}")
    draw_bounding_boxes(start_image, json_data, annotated_image_path)

    # 4. 生成文本提示
    prompt = generate_text_prompt(target_screen_description)

    # 5. 获取模型响应
    model_response = get_model_response(prompt, model, tokenizer)

    # 6. 验证响应
    is_valid, model_element = validate_response(model_response, json_data)

    # 7. 获取人工标注
    human_element = get_human_label(human_label_json)

    # 8. 计算准确率
    is_accurate = calculate_accuracy(model_element, human_element)

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

def process_dataset(dataset_dir, output_dir, model, tokenizer):
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
            elif file.startswith('hu_labeled') and file.endswith('start.json'):
                human_label_json = os.path.join(root, file)
        if start_image and end_image and yolo_json and human_label_json:
            pairs.append((start_image, end_image, yolo_json, human_label_json))

    for start_image, end_image, yolo_json, human_label_json in tqdm(pairs, desc="Processing image pairs"):
        result = process_pair(start_image, end_image, yolo_json, human_label_json, output_dir, model, tokenizer)
        results.append(result)
        total_pairs += 1
        if result['is_accurate']:
            accurate_pairs += 1

    accuracy = accurate_pairs / total_pairs if total_pairs > 0 else 0
    print(f"Overall accuracy: {accuracy:.2%}")

    return results, accuracy

def save_results_to_csv(results, accuracy, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["start_image", "end_image", "target_description", "model_response",
                      "is_valid", "suggested_element", "human_labeled_element",
                      "is_accurate", "annotated_image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

        # 在最后一行添加总体准确率
        writer.writerow({"start_image": "Overall Accuracy", "end_image": f"{accuracy:.2%}"})

def main():
    dataset_dir = "human_labeled"
    output_dir = "results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer, model = setup_model()

    # 打印模型信息
    print(f"Model device map: {model.hf_device_map}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    results, accuracy = process_dataset(dataset_dir, output_dir, model, tokenizer)

    # 保存结果到CSV文件
    csv_output_file = os.path.join(output_dir, "analysis_results.csv")
    save_results_to_csv(results, accuracy, csv_output_file)

    print(f"Analysis complete. Results saved to {csv_output_file}")
    print(f"Overall accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
import csv
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 模型设置
MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_model():
    print("Setting up the model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 使用 Accelerate 库来加载大型模型
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model = load_checkpoint_and_dispatch(
        model,
        MODEL_PATH,
        device_map="auto",
        no_split_module_classes=["CogVLMModel"],
        dtype=torch.float16
    )

    model.eval()
    return tokenizer, model

def recur_move_to(item, tgt, criterion_func):
    if criterion_func(item):
        return item.to(tgt)
    elif isinstance(item, (list, tuple)):
        return type(item)([recur_move_to(v, tgt, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
    else:
        return item

def collate_fn(features, tokenizer):
    images = [feature.pop('images', None) for feature in features if 'images' in feature]
    tokenizer.pad_token = tokenizer.eos_token
    max_length = max(len(feature['input_ids']) for feature in features)

    def pad_to_max_length(feature, max_length):
        padding_length = max_length - len(feature['input_ids'])
        feature['input_ids'] = torch.cat([feature['input_ids'], torch.full((padding_length,), tokenizer.pad_token_id)])
        feature['token_type_ids'] = torch.cat([feature['token_type_ids'], torch.zeros(padding_length, dtype=torch.long)])
        feature['attention_mask'] = torch.cat([feature['attention_mask'], torch.zeros(padding_length, dtype=torch.long)])
        if feature['labels'] is not None:
            feature['labels'] = torch.cat([feature['labels'], torch.full((padding_length,), tokenizer.pad_token_id)])
        else:
            feature['labels'] = torch.full((max_length,), tokenizer.pad_token_id)
        return feature

    features = [pad_to_max_length(feature, max_length) for feature in features]
    batch = {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0].keys()
    }

    if images:
        batch['images'] = images

    return batch

def get_image_description(image_path, model, tokenizer):
    query = 'Describe this image in detail, and the description should be between 15 to 80 words.'
    image = Image.open(image_path).convert('RGB')
    input_sample = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    input_batch = collate_fn([input_sample], tokenizer)

    # 确保所有输入都在正确的设备上
    input_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in input_batch.items()}

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": 1,
    }

    try:
        with torch.no_grad():
            outputs = model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(outputs)
    except RuntimeError as e:
        print(f"Error during generation: {e}")
        print("GPU Memory Usage:")
        print(torch.cuda.memory_summary(device=device))
        return "Error: Unable to generate description."

    return outputs[0].split("<|end_of_text|>")[0].strip()

def parse_json_file(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)

def draw_bounding_boxes(image_path, json_data, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for idx, element in enumerate(json_data["compos"], 1):
        box = element["bounding_box"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(idx), fill="red")

    image.save(output_path)
    return image

def generate_text_prompt(target_description):
    return f"What UI element should I tap to transit to the screen that contains {target_description}? Please tell me the index number of it."

def get_model_response(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def validate_response(response, json_data):
    try:
        index = int(response) - 1
        if 0 <= index < len(json_data["compos"]):
            return True, json_data["compos"][index]
        else:
            return False, None
    except ValueError:
        return False, None

def get_human_label(human_label_path):
    with open(human_label_path, 'r') as f:
        human_data = json.load(f)
    for element in human_data['compos']:
        if element.get('is_correct', False):
            return element
    return None

def calculate_accuracy(model_element, human_element):
    if not model_element or not human_element:
        return False

    def iou(box1, box2):
        # 计算两个边界框的交并比（IoU）
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 计算交集区域
        intersect_x1 = max(x1, x3)
        intersect_y1 = max(y1, y3)
        intersect_x2 = min(x2, x4)
        intersect_y2 = min(y2, y4)

        intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

        # 计算并集区域
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersect_area

        if union_area == 0:
            return 0

        return intersect_area / union_area

    # 如果IoU大于0.5，我们认为预测是准确的
    return iou(model_element['bounding_box'], human_element['bounding_box']) > 0.5

def process_pair(start_image, end_image, yolo_json, human_label_json, output_dir, model, tokenizer):
    # 1. 获取目标屏幕描述
    target_screen_description = get_image_description(end_image, model, tokenizer)

    # 2. 解析JSON文件
    json_data = parse_json_file(yolo_json)

    # 3. 绘制边界框
    annotated_image_path = os.path.join(output_dir, f"annotated_{os.path.basename(start_image)}")
    draw_bounding_boxes(start_image, json_data, annotated_image_path)

    # 4. 生成文本提示
    prompt = generate_text_prompt(target_screen_description)

    # 5. 获取模型响应
    model_response = get_model_response(prompt, model, tokenizer)

    # 6. 验证响应
    is_valid, model_element = validate_response(model_response, json_data)

    # 7. 获取人工标注
    human_element = get_human_label(human_label_json)

    # 8. 计算准确率
    is_accurate = calculate_accuracy(model_element, human_element)

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

def process_dataset(dataset_dir, output_dir, model, tokenizer):
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
            elif file.startswith('hu_labeled') and file.endswith('start.json'):
                human_label_json = os.path.join(root, file)
        if start_image and end_image and yolo_json and human_label_json:
            pairs.append((start_image, end_image, yolo_json, human_label_json))

    for start_image, end_image, yolo_json, human_label_json in tqdm(pairs, desc="Processing image pairs"):
        result = process_pair(start_image, end_image, yolo_json, human_label_json, output_dir, model, tokenizer)
        results.append(result)
        total_pairs += 1
        if result['is_accurate']:
            accurate_pairs += 1

    accuracy = accurate_pairs / total_pairs if total_pairs > 0 else 0
    print(f"Overall accuracy: {accuracy:.2%}")

    return results, accuracy

def save_results_to_csv(results, accuracy, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["start_image", "end_image", "target_description", "model_response",
                      "is_valid", "suggested_element", "human_labeled_element",
                      "is_accurate", "annotated_image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

        # 在最后一行添加总体准确率
        writer.writerow({"start_image": "Overall Accuracy", "end_image": f"{accuracy:.2%}"})

def main():
    dataset_dir = "human_labeled"
    output_dir = "results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer, model = setup_model()

    # 打印模型信息
    print(f"Model device map: {model.hf_device_map}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    results, accuracy = process_dataset(dataset_dir, output_dir, model, tokenizer)

    # 保存结果到CSV文件
    csv_output_file = os.path.join(output_dir, "analysis_results.csv")
    save_results_to_csv(results, accuracy, csv_output_file)

    print(f"Analysis complete. Results saved to {csv_output_file}")
    print(f"Overall accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
