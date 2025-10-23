import os
import random
import sqlite3
import time
from pathlib import Path
from utils.llm_tools import llm_score, llm_mutate

# --- Configuration ---
POPULATION_SIZE = 5
NUM_GENERATIONS = 3
MUTATION_RATE = 0.4
DB_PATH = "db.sqlite3"
POP_DIR = Path("population")
TARGET_TASK = "Write a Python function that returns the nth Fibonacci number efficiently."

# --- Setup ---
POP_DIR.mkdir(exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS population (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    generation INTEGER,
    score REAL,
    parent1 INTEGER,
    parent2 INTEGER
)
""")
conn.commit()

# --- Helpers ---

def random_code():
    """Create random starting code snippet."""
    templates = [
        "def fib(n): return 1 if n<=2 else fib(n-1)+fib(n-2)",
        "def fib(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
        "def fib(n): import math; return round(((1+math.sqrt(5))/2)**n/math.sqrt(5))"
    ]
    return random.choice(templates)

def save_code(code, generation, index):
    fname = f"gen_{generation}_{index}.py"
    fpath = POP_DIR / fname
    fpath.write_text(code)
    return fname

def load_code(fname):
    return (POP_DIR / fname).read_text()

def select_top(cur, gen, limit=2):
    cur.execute("SELECT id, filename, score FROM population WHERE generation=? ORDER BY score DESC LIMIT ?", (gen, limit))
    return cur.fetchall()

# --- Genetic Algorithm ---

def main():
    # Initialize population
    population = [random_code() for _ in range(POPULATION_SIZE)]
    for i, code in enumerate(population):
        fname = save_code(code, 0, i)
        score = llm_score(code, TARGET_TASK)
        cur.execute("INSERT INTO population (filename, generation, score) VALUES (?, ?, ?)", (fname, 0, score))
    conn.commit()

    # Evolve generations
    for gen in range(1, NUM_GENERATIONS + 1):
        print(f"\n=== Generation {gen} ===")

        parents = select_top(cur, gen - 1)
        if not parents:
            print("No parents found; stopping.")
            break

        next_gen = []
        for i in range(POPULATION_SIZE):
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            code1 = load_code(p1[1])
            code2 = load_code(p2[1])

            # Simple crossover: concat half of each
            midpoint = len(code1) // 2
            child_code = code1[:midpoint] + "\n" + code2[midpoint:]

            # Mutate via LLM
            if random.random() < MUTATION_RATE:
                child_code = llm_mutate(child_code, TARGET_TASK)

            fname = save_code(child_code, gen, i)
            score = llm_score(child_code, TARGET_TASK)
            cur.execute("INSERT INTO population (filename, generation, score, parent1, parent2) VALUES (?, ?, ?, ?, ?)",
                        (fname, gen, score, p1[0], p2[0]))
            conn.commit()
            print(f"→ {fname}: score={score:.2f}")

        time.sleep(1)

    print("\nDone evolving!")

if __name__ == "__main__":
    main()
