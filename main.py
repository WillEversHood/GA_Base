import os
import random
import sqlite3
import time
from pathlib import Path
from utils.llm_tools import llm_score, llm_mutate, llm_crossover

# --- Configuration ---
POPULATION_SIZE = 5
NUM_GENERATIONS = 3
MUTATION_RATE = 0.4
NUM_MUTATIONS = 2
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
    parent2 INTEGER,
    origin_id INTEGER
)
""")
conn.commit()

# --- Helpers ---


templates = [
    "def fib(n): return 1 if n<=2 else fib(n-1)+fib(n-2)",
    "def fib(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
    "def fib(n): import math; return round(((1+math.sqrt(5))/2)**n/math.sqrt(5))"
]

def save_code(code, generation, index):
    fname = f"gen_{generation}_{index}.py"
    fpath = POP_DIR / fname
    fpath.write_text(code)
    return fname

def load_code(fname):
    return (POP_DIR / fname).read_text()

def select_top(cur, gen, limit=2):
    cur.execute("SELECT id, filename, origin_id FROM population WHERE generation=? ORDER BY score DESC LIMIT ?", (gen, limit))
    return cur.fetchall()

"""integrate a parent selection function sequentially with select_top"""
def parent_selection(parents):
    parents_for_mutation = []
    for i in range(NUM_MUTATIONS):
        code = load_code(parents[i][1])
        id = parents[i][0]
        origin_id = parents[i][2]
        parents_for_mutation.append((code, id, origin_id))
    return parents_for_mutation

# --- Genetic Algorithm ---

def main():
    index = 0
    # Initialize population
    # origin_id = island #
    population = templates
    for i, code in enumerate(population):
        index += 1
        fname = save_code(code, 0, index)
        score = llm_score(code, TARGET_TASK)
        cur.execute("INSERT INTO population (filename, generation, score, origin_id) VALUES (?, ?, ?, ?)", (fname, 0, score, i))
    conn.commit()

    # Evolve generations
    for gen in range(0, NUM_GENERATIONS):
        index += 1
        print(f"\n=== Generation {gen} ===")

        parents = select_top(cur, gen - 1)
        p = parent_selection(parents)

        # place a parent selection function here
        if not p:
            print("No parents found; stopping.")
            break

            # Mutate via LLM
        if random.random() < MUTATION_RATE:
            parent_num = p[int(random.random() * (len(p)- 1))]
            child_code = llm_mutate(p[parent_num][0], TARGET_TASK)
            fname = save_code(child_code, gen, index)
            score = llm_score(child_code, TARGET_TASK)
            cur.execute("INSERT INTO population (filename, generation, score, parent1, parent2, origin_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (fname, gen, score, p[parent_num][1], 0, p[parent_num][2]))
            conn.commit()
        else:
            # update this to be any two parents selevted without same origin_id
            child_code = llm_crossover(p[0][0], p[1][0], TARGET_TASK)

            fname = save_code(child_code, gen, index)
            score = llm_score(child_code, TARGET_TASK)
            cur.execute("INSERT INTO population (filename, generation, score, parent1, parent2, origin_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (fname, gen, score, p[0][1], p[1][1], p[0][2]))
            conn.commit()
        print(f"→ {fname}: score={score:.2f}")

        time.sleep(1)

    print("\nDone evolving!")

if __name__ == "__main__":
    main()
