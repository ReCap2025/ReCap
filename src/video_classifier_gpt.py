import os
import json
import argparse
import re
import base64

import cv2
import numpy as np

from prompts import failure_identification_prompt

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "API_kEY"))

class SubTaskResult(BaseModel):
    skill_description: str
    execution_result: str
    task_succeed: bool
    
class FrameDescription(BaseModel):
    image: str
    description: str

class VideoAnalysisResult(BaseModel):
    explanation: str
    # img_description: list[FrameDescription]
    task_results: list[SubTaskResult]

def score_calculation(all_results):
    # Accuracy calculation
    total = len(all_results)
    correct = sum(1 for result in all_results if result['answer'] == result['prediction'])
    acc = correct / total if total > 0 else 0
    
    # Print the id of the false samples
    print([i for i, result in enumerate(all_results) if result['answer'] != result['prediction']])
    
    # F1 score calculation
    tp = sum(1 for result in all_results if result['answer'] == result['prediction'] and result['answer'] == 1)
    fp = sum(1 for result in all_results if result['answer'] != result['prediction'] and result['prediction'] == 1)
    fn = sum(1 for result in all_results if result['answer'] != result['prediction'] and result['answer'] == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return acc, precision, recall, f1

def extract_json_from_string(str):
    json_str = str.split('```json')[1].strip().split('```')[0].strip()
    clean_json_string = '\n'.join([line.split('//')[0].strip() for line in json_str.splitlines()])
    output = json.loads(clean_json_string)
    return output

def encode_video(path):
    video = cv2.VideoCapture(path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    print(f"{len(base64Frames)} frames read from {path}")
    return base64Frames

def analyze_video(video_file, generated_plan, prompt, args, additional_file=None):
    prompt = prompt.replace('<GOAL>', args.goal.replace('_', ' '))
    prompt = prompt.replace('<LANGUAGE_PLAN>', str(generated_plan))
    
    # Sample 20 frames from the video for analysis and include the last one
    indices = np.linspace(0, len(video_file) - 1, 20, dtype=int, endpoint=True).tolist()
    
    video_file = [video_file[i] for i in indices]
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 256}, video_file)
            ],
        },
    ]

    # Make the LLM request.
    print("Making LLM inference request...")
    params = {
        "model": "gpt-4.1",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1024,
        "response_format": VideoAnalysisResult
    }
    
    result = client.beta.chat.completions.parse(**params)
    analysis_result = result.choices[0].message.content

    return analysis_result

def safe_json_load(raw_str):
    """
    Extract and parse a valid JSON object from a raw string that may contain extra text
    (e.g., markdown formatting or log prefixes).
    Raises ValueError if parsing fails.
    """
    match = re.search(r"\{.*\}", raw_str, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        raise ValueError("No valid JSON object found in the input string.")

def main(args):
    data_root = args.data_root
    task_name = args.task_name
    
    task_dir = os.path.join(data_root, task_name)
    output_dir = os.path.join(task_dir, 'results')
    recorded_dir = os.path.join(task_dir, 'videos')
    languages_dir = os.path.join(task_dir, 'languages')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get recorded results
    recorded_files = os.listdir(recorded_dir)
    recorded_files = sorted(recorded_files, key=lambda x: int(x.split('.')[0].split('_')[0]))
    
    all_results = []
    
    system_prompt = failure_identification_prompt
    
    # Iterate over files from episode0 to episodeN
    N = 5
    for i in range(N):
        output_file = os.path.join(output_dir, f"episode{i}.json")
        if os.path.exists(output_file):
            print(f"Results for episode{i} already exist, skipping...")
            pred_result = json.load(open(output_file))
            all_results.append(pred_result)
            continue
        
        # Process the video
        filename = recorded_files[i]
        ground_truth_success = True if 'True' in filename else False
        results = {}
        
        filepath = os.path.join(recorded_dir, filename)

        print(f"Processing {filename}...")
        video_file = encode_video(filepath)
        
        # Load the generated plan
        language_file = os.path.join(languages_dir, f"episode{i}.json")
        with open(language_file, 'r') as f:
            language_data = json.load(f)
        generated_plan = list(language_data.keys())
        
        success_flag = False
        while not success_flag:
            try:
                pred_result = analyze_video(video_file, generated_plan, system_prompt, args, additional_file=None)
                success_flag = True
                # breakpoint()
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print("Retrying...")

        pred_result = safe_json_load(pred_result)
        results['answer'] = ground_truth_success
        results['response'] = pred_result
        results['prediction'] = all(task['task_succeed'] for task in pred_result['task_results'])
        
        # Write to output JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        all_results.append(results)
        print(f"Results saved to {output_file}")
    
    # Calculate accuracy
    acc, pre, rec, f1 = score_calculation(all_results)
    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze robot performance in videos using GPT-4.1.")
    parser.add_argument('--data_root', type=str, default='./demo')
    parser.add_argument('--task_name', type=str, default='KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it')
    parser.add_argument('--goal', type=str, default='turn_on_the_stove_and_put_the_moka_pot_on_it')

    args = parser.parse_args()
    main(args)