from textwrap import dedent
from openai import OpenAI 
import os
from dotenv import load_dotenv
import re
import json
import numpy as np
load_dotenv()

# --- Replace these with actual API calls to Gemini ---
# Example:
# from google import genai
# model = genai.GenerativeModel("gemini-1.5-pro")
class LLMTools:
    def __init__(self):
        self.MONEY = []

    def cost_calc(self, tokens: int)-> float:
        # Example cost calculation for Gemini API usage
        cost_per_1k_tokens = 0.0001  # hypothetical cost
        cost = (tokens / 1000) * cost_per_1k_tokens
        return cost

    def llm_score(self, code: str, task: str) -> float:

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file")
        """
        Use an LLM to intelligently mutate the code while preserving function.
        """
        prompt = f"""You are an expert software engineer and code evaluator. Your task is to assess how well a given piece of code accomplishes a specified goal using a detailed rubric.

### Instructions:
1.  Read the `Task` description carefully.
2.  Analyze the provided `Code` to understand its approach.
3.  Evaluate the code against the `Rubric (100-point scale)` provided below.
4.  **Step 1: Reasoning.** First, you MUST provide a step-by-step evaluation inside a <reasoning> XML block. For each criterion in the rubric, justify the score you are assigning.
5.  **Step 2: JSON Output.** Output a single JSON object with one key value pare "score : ____ ". Start this block with <```json> and end with ```> to ensure proper formatting.

---

### Rubric (270-point scale)

**1. Correctness (60 points)**
    * **Task Fulfillment (30 pts):** Does the code produce the correct output for the primary task? (30 = perfect; 15 = partially correct; 0 = incorrect)
    * **Constraint Adherence (20 pts):** Does the code adhere to all constraints mentioned in the task (e.g., "must use recursion," "time limit," "no external libraries")? (20 = all constraints met; 0 = one or more constraints failed)
    * **Error Handling (10 pts):** Does the code handle common edge cases (e.g., empty inputs, invalid types, divide-by-zero) gracefully? (10 = robust; 5 = handles some cases; 0 = crashes on edge cases)

**2. Efficiency (150 points)**
    * **Algorithmic Complexity (75 pts):** Does the code use an efficient algorithm for this task? (e.g., uses a hash map for O(1) lookups instead of an O(n) list search; uses O(n log n) sort vs O(n^2)). (60 = optimal; 30 = standard but sub-optimal; 0 = brute-force/highly inefficient)
    * **Code-Level Efficiency (75 pts):** Does the code avoid redundant operations (e.g., re-calculating a value inside a loop when it could be done once outside)? (60 = efficient; 0 = redundant work)

**3. Readability & Style (60 points)**
    * **Clarity (30 pts):** Is the code well-structured with clear variable names (e.g., `user_index` instead of `i`)?
    * **Documentation (30 pts):** Are there comments or docstrings explaining *why* the code does what it does, especially for complex parts?

---

### Task:
{task}

---

### Code:
```python
{code}"""

        client = OpenAI(
        api_key= api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        response = client.chat.completions.create(
        model="gemini-2.5-flash-lite", # or other Gemini models
        messages=[
            {"role": "system", "content": "You are an exacting code evaluator."},
            {"role": "user", "content": prompt}
        ],
        #max_output_tokens=100,
        temperature=0,
        )
        tokens = response.usage.total_tokens
        cost = self.cost_calc(tokens)
        self.MONEY.append(np.array([tokens, cost]))
        if response and response.choices:
            jobject = response.choices[0].message.content.strip()
            print(jobject)
            match = re.search(r'</reasoning>\s*```json\s*(\{.*?\})\s*```', jobject, re.DOTALL)
            score = int(json.loads(match.group(1))['score'])
            

            '''print("==================================================")
            print(code)
            print("--------------------------------------------------")
            print(score)'''
            return score
        else:
            return "[Error] No response content."





    def llm_mutate(self, code: list, task: str, context_limit: int) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not code:
            raise ValueError("Code list is empty")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file")
        """
        Use an LLM to intelligently mutate the code while preserving function.
        """
        #process list of code examples into single 
        if len(code) > context_limit:
            code = "\n\n###\n\n".join(code[:context_limit])
        else:
            code = "\n\n###\n\n".join(code)
        prompt = dedent(f"""
        Improve the following code to better accomplish the
        Task: {task}
        Original code examples:
        {code}

        Return only the improved code and no explanations at all.    
        """)
        # Simulated "mutation"
        client = OpenAI(
        api_key= api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        response = client.chat.completions.create(
        model="gemini-2.5-flash-lite", # or other Gemini models
            messages=[
                {"role": "system", "content": "You are a talented software engineer."},
                {"role": "user", "content": prompt}
            ],
            #max_output_tokens=100,
        temperature=0.7,
        )
        # cost calculation
        tokens = response.usage.total_tokens
        cost = self.cost_calc(tokens)
        self.MONEY.append(np.array([tokens, cost]))
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




    def llm_crossover(self, code1: str, code2: str, task: str) -> str:
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
            model="gemini-2.5-flash-lite", # or other Gemini models
            messages=[
                {"role": "system", "content": "You are a talented software engineer."},
                {"role": "user", "content": prompt}
            ],
            #max_output_tokens=100,
            temperature=0.7,
        )
        tokens = response.usage.total_tokens
        cost = self.cost_calc(tokens)
        self.MONEY.append(np.array([tokens, cost]))
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