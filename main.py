import os
import random
import sqlite3
import time
from pathlib import Path
from utils.llm_tools import LLMTools
import numpy as np
from visualizations import Visualizations

# --- Configuration ---
POPULATION_SIZE = 5
NUM_GENERATIONS = 2
MUTATION_RATE = 0.4
NUM_MUTATIONS = 2
DB_PATH = "db.sqlite3"
POP_DIR = Path("population")
TARGET_TASK = "Write a Python function that returns the nth Fibonacci number efficiently."
PERFORMANCE_TRACKING = np.zeros((NUM_GENERATIONS, NUM_MUTATIONS))

# --- Setup ---
def db_setup():
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
    return conn, cur

# --- Helpers ---


templates = [
    "def fib(n): return 1 if n<=2 else fib(n-1)+fib(n-2)"#,
   # "def fib(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
   # "def fib(n): import math; return round(((1+math.sqrt(5))/2)**n/math.sqrt(5))"
]

def save_code(code, generation, index):
    fname = f"gen_{generation}_{index}.py"
    fpath = POP_DIR / fname
    fpath.write_text(code)
    return fname

def load_code(fname):
    return (POP_DIR / fname).read_text()

def select_top(cur, gen):
    cur.execute("SELECT id, filename, origin_id, score FROM population WHERE generation=? ORDER BY score DESC", (gen,))
    return cur.fetchall()

"""integrate a parent selection function sequentially with select_top"""
def parent_selection(parents, gen):
    parents_for_mutation = []
    distinct_origins = {t[2] for t in parents}
    count = len(distinct_origins)
    parent_array = np.array(parents)
    print("Parent array:")
    print(parent_array)
    print(parent_array.shape)
    for island in range(count-1):
        mask = (parent_array[:, 2].astype(int) == island)
        filtered_array = parent_array[mask,:]
        print('filtered array:')
        print(filtered_array)
        # parent selection logic here add power law etc.
        id = filtered_array[0,0]  # select top 1 from each island
        print('id')
        print(id)
        name = str(filtered_array[0,1])
        print('name')
        print(name)
        origin_id = int(filtered_array[0,2])
        print('origin_id')
        print(origin_id)
        code = load_code(name)
        parents_for_mutation.append((code, id, origin_id))
    return parents_for_mutation

# --- Genetic Algorithm ---

def main():
    llmTools = LLMTools()
    conn, cur = db_setup()
    index = 0
    # Initialize population
    # origin_id = island #
    
    population = templates
    for i, code in enumerate(population):
        index += 1
        fname = save_code(code, 0, index)
        score = llmTools.llm_score(code, TARGET_TASK)
        cur.execute("INSERT INTO population (filename, generation, score, origin_id) VALUES (?, ?, ?, ?)", (fname, 0, score, i))
    conn.commit()
    
    '''Initialize performance tracking structure here'''
    vis = Visualizations(NUM_GENERATIONS)

    # Evolve generations
    for gen in range(1, NUM_GENERATIONS):
        
        print(f"\n=== Generation {gen} ===")

        parents = select_top(cur, gen - 1)
        p = parent_selection(parents, gen - 1)

        # place a parent selection function here
        if not p:
            print("No parents found; stopping.")
            break

            # Mutate via LLM
        # performance array 3xNUM_MUTATIONS
        perf_array = np.zeros((3, NUM_MUTATIONS))   
        for k in range(NUM_MUTATIONS):
            print(f"k: {k} --")
            index += 1
            # for randum island assignment and parent selection so it outside of mutation bracket :)          
            parent_num = int(random.random() * (len(p)- 1))
            if random.random() < MUTATION_RATE:
                child_code = llmTools.llm_mutate(p[parent_num][0], TARGET_TASK)
                fname = save_code(child_code, gen, index)
                score = llmTools.llm_score(child_code, TARGET_TASK)
                cur.execute("INSERT INTO population (filename, generation, score, parent1, parent2, origin_id) VALUES (?, ?, ?, ?, ?, ?)",
                            (fname, gen, score, p[parent_num][1], 0, p[parent_num][2]))
                conn.commit()
            else:
                # update this to be any two parents selected without same origin_id
                child_code = llmTools.llm_crossover(p[0][0], p[1][0], TARGET_TASK)

                fname = save_code(child_code, gen, index)
                score = llmTools.llm_score(child_code, TARGET_TASK)
                cur.execute("INSERT INTO population (filename, generation, score, parent1, parent2, origin_id) VALUES (?, ?, ?, ?, ?, ?)",
                            (fname, gen, score, p[0][1], p[1][1], p[0][2]))
                conn.commit()
            print(f"→ {fname}: score={score:.2f}")

            # Track Performance
            perf_array[0][k] = score
            perf_array[1][k] = index
            perf_array[2][k] = p[parent_num][2]  # origin_id            
        #vis.log_performance(perf_array)
        #vis.plot_performance()
        time.sleep(1)
        # count the cost
        money_array = np.array(llmTools.MONEY)
        money_sums = money_array.sum(axis=0)
        with open('costs.txt', 'a', encoding='utf-8') as f:
            f.write(f" {money_sums[0]}, ${money_sums[1]}")
    print("\nDone evolving!")
    
if __name__ == "__main__":
    main()
    #print(select_top(db_setup()[1], 0))
