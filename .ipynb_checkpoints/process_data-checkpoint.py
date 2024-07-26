import json
from PIL import Image, ImageDraw, ImageFont
import random
import os
import shutil

def parse_json_file(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)

def get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_bounding_boxes(image_path, json_data, font_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.truetype(font_path, 20)
    labels = []

    # 计算所有元素的面积，保持原始顺序
    elements = []
    for idx, element in enumerate(json_data["compos"], 1):
        position = element
        column_min = position.get("column_min", 0)
        row_min = position.get("row_min", 0)
        column_max = position.get("column_max", 0)
        row_max = position.get("row_max", 0)
        area = (column_max - column_min) * (row_max - row_min)
        elements.append((idx, element, area))

    # 如果元素数量大于等于6，则找出面积最小的三个的索引
    # if len(elements) >= 6:
    #     sorted_elements = sorted(elements, key=lambda x: x[2])
    #     indices_to_remove = set([e[0] for e in sorted_elements[:3]])
    # else:
    #     indices_to_remove = set()
    
    indices_to_remove = set()

    for idx, element, area in elements:
        if idx in indices_to_remove:
            continue

        position = element
        column_min = position.get("column_min", 0)
        row_min = position.get("row_min", 0)
        column_max = position.get("column_max", 0)
        row_max = position.get("row_max", 0)
        
        # 只标记面积大于等于 20 的元素
        # if area >= 20:
        
        box = (column_min, row_min, column_max, row_max)
        color = get_random_color()

        # 创建一个新的 Image 对象作为覆盖层
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # 绘制虚线边框
        dash_length = 5
        for i in range(0, int(2*(column_max-column_min + row_max-row_min)), dash_length*2):
            if i < 2*(column_max-column_min):
                start = (column_min + i//2, row_min if i % (4*dash_length) < 2*dash_length else row_max)
                end = (min(column_min + (i+dash_length)//2, column_max), row_min if i % (4*dash_length) < 2*dash_length else row_max)
            else:
                i -= 2*(column_max-column_min)
                start = (column_min if i % (4*dash_length) < 2*dash_length else column_max, row_min + i//2)
                end = (column_min if i % (4*dash_length) < 2*dash_length else column_max, min(row_min + (i+dash_length)//2, row_max))
            overlay_draw.line([start, end], fill=color, width=2)

        # 绘制标签
        text = str(idx)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_position = (column_min, max(0, row_min - text_height - 5))

        # 绘制标签背景
        overlay_draw.rectangle((label_position[0], label_position[1] + 5, 
                                label_position[0] + text_width, 
                                label_position[1] + text_height + 5), 
                               fill=color)

        # 绘制标签文本
        overlay_draw.text(label_position, text, fill="white", font=font)

        # 将覆盖层合并到原图
        image = Image.alpha_composite(image.convert('RGBA'), overlay)
        

    return image.convert('RGB')

def preprocess_images(dataset_dir, output_dir, font_path):
    print("Pre-processing images...")
    
    for root, _, files in os.walk(dataset_dir):
        if root == dataset_dir:
            continue
        start_image = yolo_json = None
        save_path = os.path.join(output_dir, os.path.basename(root))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in files:
            if file.endswith('start.jpg'):
                start_image_name = file
                start_image = os.path.join(root, file)
            elif file.endswith('end.jpg'):
                end_image_name = file
                end_image = os.path.join(root, file)
            elif file.startswith('YOLO') and file.endswith('start.json'):
                yolo_name = file
                yolo_json = os.path.join(root, file)
        
        if start_image and yolo_json:
            json_data = parse_json_file(yolo_json)
            annotated_image = draw_bounding_boxes(start_image, json_data, font_path)
            annotated_image.save(os.path.join(save_path, start_image_name))
            
            shutil.copy2(yolo_json, os.path.join(save_path, yolo_name))
            shutil.copy2(end_image, os.path.join(save_path, end_image_name))

def main():
    dataset_dir = "data/human_labeled"
    output_dir = "process_data"
    font_path = "Roboto-Bold.ttf"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Pre-process images
    preprocess_images(dataset_dir, output_dir, font_path)

if __name__ == "__main__":
    main()