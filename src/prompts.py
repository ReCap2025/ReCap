failure_identification_prompt = '''Analyze the following video to identify the actions in the video are consistent with the following language plan.
This is a recorded video that the robot performed a task in a simulated environment according to the predicted language plan.
However, the robot may not have completed the task successfully.
Your task is to determine if each action in the video corresponds to the language plan completely.

The goal is: <GOAL>.
The predicted language plan is: <LANGUAGE_PLAN>.

Provide your explanation following the steps:
1. Find the sub-goals required to achieve the main goal.
2. Identify if each sub-goal has been fulfilled.
3. If any of the sub-goals are not fulfilled, identify the actions in the video that correspond to each sub-goal.
4. Identify the reasons for the success or failure of each language instruction in the language plan.

Based on the explanation, please provide the following information:
skill_description: The skill instruction in the language plan.
execution_result: The result of the action in the video.
task_succeed: Whether the action in the video is successful or not.

Additional hints:
<ADDITIONAL_HINTS>
'''

language_plan_prompt = '''
Analyze the following video to identify the robot's actions in the video are consistent with language goal.
This is a recorded video that the robot performed a task in a simulated environment according to the predicted language plan.
However, the robot may not have completed the task successfully.
Sometimes the generated plan may not be correct.
Your task is to determine at which step the robot failed to follow the language plan.
The goal is: <GOAL>.
The predicted language plan is: <LANGUAGE_PLAN>.
Provide your explanation following the steps:
1. Find the sub-goals required to achieve the main goal.
2. Identify if each sub-goal has been fulfilled.
3. If any of the sub-goals are not fulfilled, identify the actions in the video that correspond to each sub-goal.
4. Identify the reasons for the success or failure of each language instruction in the language plan.
'''

language_replan_prompt = '''
Baed on the analysis of the video, your task is to regenerate a different language plan.
According to failure analysis, the robot may not have completed the task successfully.
The goal is: <GOAL>.\n
The predicted language plan is: <LANGUAGE_PLAN>.\n
The history of the video analysis is: <HISTORY>.\n

After analyzing the video, please regenerate the language plan from the first failed one.
The language plan should be a list of instructions that the robot should follow to complete the task successfully.
'''