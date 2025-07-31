# ReCap

This README provides instructions for setting up and running video analysis demo for the ReCap framework.

## Installation

To install the necessary dependencies, please run the following commands in your terminal:

```bash
pip install -r requirements.txt
pip install -q -U google-genai
```

## API Key Configuration

Before running the scripts, you need to replace the placeholder API keys for OpenAI and Gemini. Locate and update the API keys in the following files:

* `./src/video_classifier_gpt.py`
* `./src/refine_language_plan_gemini.py`

## Running the Applications

### Video Failure Identification

To run the video failure identification process on five sample videos, execute the following command:

```bash
python ./src/video_classifier_gpt.py
```

Example outputs for this task can be found under `./demo/video_failure_identification/results`.

### Language Plan Strengthening

To run the language plan strengthening process, use the following command:

```bash
python ./src/refine_language_plan_gemini.py
```

Example outputs for this task can be found under `./demo/language_plan_strengthening/results`.
