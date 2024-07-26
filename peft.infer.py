import torch
from torch.nn.parallel import DataParallel
from PIL import Image

def setup_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = DataParallel(model)  # 使用 DataParallel 包装模型
    return model

def run_inference(model, tokenizer, image, query, gen_config):
    """
    执行推理操作
    
    :param model: 预训练的模型
    :param tokenizer: 分词器
    :param image: PIL Image 对象
    :param query: 查询字符串
    :param gen_config: 生成配置
    :return: 生成的响应文本
    """
    inputs = model.module.build_conversation_input_ids(tokenizer, query=query, images=[image])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def process_image_pair(model, tokenizer, start_image, end_image, description_config, target_config):
    """
    处理一对图像（起始图像和目标图像）
    
    :param model: 预训练的模型
    :param tokenizer: 分词器
    :param start_image: 起始图像（PIL Image 对象）
    :param end_image: 目标图像（PIL Image 对象）
    :param description_config: 描述生成的配置
    :param target_config: 目标预测的配置
    :return: 处理结果的字典
    """
    # 生成目标屏幕描述
    target_description = run_inference(model, tokenizer, end_image, 
                                       'Describe this image in detail, and the description should be between 15 to 80 words.',
                                       description_config)
    
    # 获取模型对于目标屏幕的响应
    model_response = run_inference(model, tokenizer, start_image,
                                   f"What UI element should I tap to transit to the screen that contains {target_description}? Please tell me the index number of it.",
                                   target_config)
    
    # 这里可以添加其他处理逻辑，比如验证响应、计算准确性等
    
    result = {
        "target_description": target_description,
        "model_response": model_response,
        # 可以在这里添加其他结果字段
    }
    
    return result

def main(model_path, image_pairs, description_config, target_config):
    model = setup_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    results = []
    for start_image, end_image in image_pairs:
        result = process_image_pair(model, tokenizer, start_image, end_image, description_config, target_config)
        results.append(result)
    
    return results

# 使用示例
if __name__ == "__main__":
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"
    
    # 假设图像对已经准备好
    image_pairs = [
        (Image.open("start_image1.jpg"), Image.open("end_image1.jpg")),
        (Image.open("start_image2.jpg"), Image.open("end_image2.jpg")),
        # 更多图像对...
    ]
    
    description_config = {
        "do_sample": True,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.7,
        "max_new_tokens": 150,
        "repetition_penalty": 1.2,
        "min_new_tokens": 30
    }
    
    target_config = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 5,
        "repetition_penalty": 1.1,
        "min_new_tokens": 1,
        "max_new_tokens": 50
    }
    
    results = main(model_path, image_pairs, description_config, target_config)
    
    # 处理结果...
    for result in results:
        print(result)