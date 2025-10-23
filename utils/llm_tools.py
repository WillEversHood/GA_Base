import random
from textwrap import dedent
from google import genai
import os
from dotenv import load_dotenv

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
    - 1.0 → fully accomplishes the task with no significant flaws
    - 0.7 → mostly works but has partial or edge-case issues
    - 0.4 → some attempt but fails key parts of the task
    - 0.0 → does not attempt or accomplish the task at all

    ### Task:
    {task_description} 

    ### Code:
    ```python
    {code_to_evaluate}"""

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"
    response = client.generate_text(
        model=model,
        prompt=prompt,
        max_output_tokens=800,
        temperature=0,
    )
    return response




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
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"
    response = client.generate_text(
        model=model,
        prompt=prompt,
        #max_output_tokens=800,
        temperature=0.7,
    )
    return response




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
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"
    response = client.generate_text(
        model=model,
        prompt=prompt,
        #max_output_tokens=800,
        temperature=0.7,
    )
    return response