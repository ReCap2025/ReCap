import os
import json
import argparse
import re
import time

from prompts import *

from google import genai
from pydantic import BaseModel

GOOGLE_API_KEY='API_kEY'

client = genai.Client(api_key=GOOGLE_API_KEY)


class TaskResult(BaseModel):
    skill: str
    execution_result: str
    task_succeed: bool
    
class FrameDescription(BaseModel):
    image: str
    description: str

class VideoAnalysisResult(BaseModel):
    explanation: str
    task_results: list[TaskResult]
    
class RegeneratedPlan(BaseModel):
    explanation: str
    goal: str
    language_plan: list[str]

def extract_json_from_string(str):
    json_str = str.split('```json')[1].strip().split('```')[0].strip()
    clean_json_string = '\n'.join([line.split('//')[0].strip() for line in json_str.splitlines()])
    output = json.loads(clean_json_string)
    return output

def encode_video(path):
    print(f"Uploading file...")
    video_file = client.files.upload(file=path)
    print(f"Completed upload: {video_file.uri}")

    # Check whether the file is ready to be used.
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    # return video_file.uri
    return video_file

def request(prompt, schema=None):  
    # Prepare the prompt message for analysis

    # Make the LLM request.
    print("Making LLM inference request...")
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[prompt],
        config={
            'response_mime_type': 'application/json',
            'response_schema': schema,
        }
    )
    return response.text

def request_video(video_file, prompt, schema=None):
    # Prepare the prompt message for analysis
    # Make the LLM request.
    print("Making LLM inference request...")
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[video_file, prompt],
        config={
            'response_mime_type': 'application/json',
            'response_schema': schema,
        }
    )
    return response.text

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

def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def inference(video_file, prompt, replace_dict=None, schema=None, additional_file=None, text_only=False):
    """
    Perform inference on the video file using the generated plan and system prompt.
    """    
    for key, value in replace_dict.items():
        prompt = prompt.replace(key, value)

    success_flag = False
    while not success_flag:
        try:
            if not text_only:
                pred_result = request_video(video_file, prompt, schema=schema, additional_file=additional_file)
                success_flag = True
            else:
                pred_result = request(prompt, schema=schema, additional_file=additional_file)
                success_flag = True
            
        except Exception as e:
            print(f"Error processing: {e}")
            print("Retrying...")

    pred_result = safe_json_load(pred_result)
    return pred_result

def main(args):
    data_root = args.data_root
    task_name = args.task_name
    task_goal = args.goal
    sample_id = args.id
    
    # task_dir = os.path.join(data_root, task_name)
    task_dir = os.path.join(data_root, task_name)
    output_dir = os.path.join(task_dir, 'results')
    recorded_dir = os.path.join(task_dir, 'videos')
    languages_dir = os.path.join(task_dir, 'languages')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples
    video_path = os.path.join(recorded_dir, f'{sample_id}_False.mp4')
    language_file_path = os.path.join(languages_dir, f'episode{sample_id}.json')
    
    output_file = os.path.join(output_dir, f"episode{sample_id}.json")
    if os.path.exists(output_file):
        print(f"Results for episode{sample_id} already exist, skipping...")
        pred_result = json.load(open(output_file))
    else:
        result = {}
        # Process the video
        print(f"Processing {video_path}...")
        video_file = encode_video(video_path)
        
        # Load the generated plan
        language_data = load_json(language_file_path)
        generated_plan = list(language_data.keys())
        
        # Failure identification
        system_prompt = language_plan_prompt
        replace_dict = {
            '<GOAL>': task_goal,
            '<LANGUAGE_PLAN>': str(generated_plan),
        }
        pred_result = inference(video_file, system_prompt, replace_dict=replace_dict, schema=VideoAnalysisResult)
        result['stage1'] = pred_result
        print(pred_result)

        # Regenerate language plan
        system_prompt = language_replan_prompt
        replace_dict = {
            '<HISTORY>': str(pred_result),
            '<GOAL>': task_goal,
            '<LANGUAGE_PLAN>': str(generated_plan),
        }
        pred_result = inference(video_file, system_prompt, replace_dict=replace_dict, schema=RegeneratedPlan)
        result['stage2'] = pred_result
        print(pred_result)
        
        # Write to output JSON file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze robot performance in videos using Gemini-2.5.")
    parser.add_argument('--data_root', type=str, default='./demo')
    parser.add_argument('--task_name', type=str, default='LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate')
    parser.add_argument('--goal', type=str, default='put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate')
    parser.add_argument('--id', type=str, default='28')

    args = parser.parse_args()
    main(args)
