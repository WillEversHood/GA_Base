from textwrap import dedent
from openai import OpenAI 
import os
from dotenv import load_dotenv
import re
import json
load_dotenv()

# --- Replace these with actual API calls to Gemini ---
# Example:
# from google import genai
# model = genai.GenerativeModel("gemini-1.5-pro")

def llm_score(code: str, task: str) -> float:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")
    """
    Use an LLM to intelligently mutate the code while preserving function.
    """
    prompt = """You are an expert software engineer and code evaluator. 
    Your task is to assess how well a given piece of code accomplishes a specified goal.

    ### Instructions:
    1. Read the task description carefully.
    2. Analyze the code provided.
    3. Decide whether the code fulfills the task’s requirements.
    4. Consider correctness, completeness, and robustness.
    5. Output ONLY a JSON object with one key: "score", whose value is a float between 0 and 1.

    ### Example Scoring:
    - 300 → Perfect cutting edge implementation that is faster the best known algorithm for this task
    - 150 → Semi Correct implementation that attempts novel and advanced techniques
    - 100 → Correct implementation that attempts standard efficient techniques
    - 70 → Correct and efficient implementation
    - 30 → Incomplete or incorrect implementation that doesn't address the task

    ### Task:
    {task_description} 

    ### Code:
    ```python
    {code_to_evaluate}"""

    client = OpenAI(
        api_key= api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash", # or other Gemini models
        messages=[
            {"role": "system", "content": "You are an exacting code evaluator."},
            {"role": "user", "content": prompt}
        ],
        #max_output_tokens=100,
        temperature=0,
    )

    if response and response.choices:
        jobject = response.choices[0].message.content.strip()
        match = match = re.search(r"```json\s*(.*?)\s*```", jobject, re.DOTALL)
        score = int(json.loads(match.group(1))['score'])
        '''print("==================================================")
        print(code)
        print("--------------------------------------------------")
        print(score)'''
        return score
    else:
        return "[Error] No response content."





def llm_mutate(code: str, task: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")
    """
    Use an LLM to intelligently mutate the code while preserving function.
    """
    prompt = dedent(f"""
    Improve the following code to better accomplish the
    Task: {task}
    Original code:
    {code}

    Return only the improved code and no explanations at all.    
    """)
    # Simulated "mutation"
    client = OpenAI(
        api_key= api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-pro", # or other Gemini models
        messages=[
            {"role": "system", "content": "You are a talented software engineer."},
            {"role": "user", "content": prompt}
        ],
        #max_output_tokens=100,
        temperature=0.7,
    )
    if response and response.choices:
        response_str = response.choices[0].message.content.strip()
        pattern = r"```python\s+(.*?)```"
        matches = re.search(pattern, response_str, re.DOTALL)
        if matches:
            return matches.group(1)
        else:
            print("No code block found; returning full response.")
            return response_str
    else:
        return "[Error] No response content."




def llm_crossover(code1: str, code2: str, task: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env file")
    """
    Use an LLM to intelligently combine two pieces of code.
    """
    prompt = dedent(f"""
    Combine the following two pieces of code to better accomplish the
    Task: {task}
    Code 1:
    {code1}

    Code 2:
    {code2}

    Return only the combined code and no explanations at all.    
    """)
    # Simulated "crossover"
    client = OpenAI(
        api_key= api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    response = client.chat.completions.create(
        model="gemini-2.5-pro", # or other Gemini models
        messages=[
            {"role": "system", "content": "You are a talented software engineer."},
            {"role": "user", "content": prompt}
        ],
        #max_output_tokens=100,
        temperature=0.7,
    )
    if response and response.choices:
        response_str = response.choices[0].message.content.strip()
        pattern = r"```python\s+(.*?)```"
        matches = re.search(pattern, response_str, re.DOTALL)
        if matches:
            return matches.group(1)
        else:
            print("No code block found; returning full response.")
            return response_str
    else:
        return "[Error] No response content."