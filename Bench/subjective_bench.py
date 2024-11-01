import sys
import os

# 将父目录添加到 sys.path
parent_path = os.path.dirname(sys.path[0])
# 如果父目录不在 sys.path 里则添加
if parent_path not in sys.path:
    sys.path.append(parent_path)

from Models.openai_gpt4 import OpenaiAPI
from bench_function import export_distribute_json, export_union_json
import json
import argparse

from dotenv import load_dotenv
# 加载.env文件中的环境变量
load_dotenv()

# 生成模型基准测试的json文件的主要功能
if __name__ == "__main__":
    # 读取Obj_Prompt.json 文件
    with open("Obj_Prompt.json", "r", encoding='utf-8') as f:
        data = json.load(f)['examples']
    f.close()


    openai_api_key = os.getenv("openai_api_key")
    base_url = os.getenv("base_url")
    model_name = os.getenv("model_name")

    model_api = OpenaiAPI([openai_api_key], model_name=model_name)

    for i in range(len(data)):
        directory = "../Data/Subjective_Questions"

        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print(keyword)
        print(question_type)

        export_distribute_json(
            model_api, 
            model_name, 
            directory, 
            keyword, 
            zero_shot_prompt_text, 
            question_type, 
            parallel_num=1, 
        )

        export_union_json(
            directory, 
            model_name, 
            keyword,
            zero_shot_prompt_text,
            question_type
        )
