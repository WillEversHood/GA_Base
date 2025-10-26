import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import sqlite3

class Vsualizations:

    def __init__(self):
        #self.mutations = mutations
        self.perf_data = []
        # column1 : score
        # column2 : id
        # column3 : origin_id

    def log_performance(self, score_list):
        # score list should contiain the node id, origin_id, and score
        self.perf_data.append(score_list)

    def plot_performance(self):
        performance_array = np.array(self.perf_data)
        x = np.arange(len(performance_array[:,0,0]))
        plt.figure(figsize=(10, 6)) # Optional: set the figure size

        # Extract data for each line
        y1 = np.zeros(x.shape)  # Maximum_Score
        y2 = np.zeros(x.shape)  # Average_score
        #y3 = np.zeros(x.shape)  # origin_id
        for i in range(len(x)):
            y1[i] = (performance_array[i,0,:]).max()  # Maximum_Score
            y2[i] = (performance_array[i,0,:]).mean() # Average_score
            

        # Plot the first line Maximum_Score (Red)
        plt.plot(x, y1, color='red', linestyle='-', label='Maximum Score')

        # Plot the second line Average_score (Blue)
        plt.plot(x, y2, color='blue', linestyle='--', label='Average Score')

        # Plot the third line (Green)
        #plt.plot(x, y3, color='green', linestyle=':', label='Product Wave (Green)')

        # 3. Add Labels, Title, and Legend (Best Practices)
        plt.title('Maximum and Average Scores Per Generation')
        plt.xlabel('Generations')
        plt.ylabel('Heuristic Score')
        plt.grid(True)
        plt.legend() # Displays the labels for each line

        # 4. Show the Plot
        plt.show()
        return


    def get_best_entries(self):
        """
        Connects to the SQLite database and retrieves the entry with the
        highest score for each unique combination of generation and origin_id.
        """
        conn = None
        try:
            # 1. Connect to the SQLite database
            conn = sqlite3.connect('db.sqlite3')
            cursor = conn.cursor()
            # 2. SQL Query using ROW_NUMBER()
            # This function partitions the data by (origin_id, generation) and
            # orders each partition by score, assigning 'rnk=1' to the highest score.
            sql_query = """
            SELECT
                id,              -- Include the primary key or other unique identifier
                generation,
                origin_id,
                score
            FROM
                (
                    SELECT
                        *,
                        ROW_NUMBER() OVER(
                            PARTITION BY origin_id, generation
                            ORDER BY score DESC
                        ) as rnk
                    FROM
                        population  -- Replace 'entries' with your actual table name
                )
            WHERE
                rnk = 1;
            """

            # 3. Execute the query and fetch results into a Pandas DataFrame
            #    (Pandas is excellent for viewing structured data)
            df_best_entries = pd.read_sql_query(sql_query, conn)
            
            return df_best_entries

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame() # Return empty DataFrame on error
            
        finally:
            # 4. Ensure the connection is closed
            if conn:
                #cursor.execute("DELETE FROM population")
                #conn.commit()
                conn.close()
    
    def island_visualization(self):
        performance_array = np.array(self.perf_data)
        x = np.arange(len(performance_array[:,0,0]))
        data = self.get_best_entries()
        num_islands = data['origin_id'].nunique()
        num_gen = data['generation'].nunique()
        lines = np.zeros((num_islands, num_gen))
        for i in range(num_islands):
            island = data.loc[data['origin_id'] == i]
            for j in range(num_gen):
                island_gen = island.loc[island['generation'] == j]
                if not island_gen.empty:
                    lines[i][j] = island_gen['score'].values[0]
                elif j > 0:
                    lines[i][j] = lines[i][j-1]  # carry forward previous score if no entry
                else:
                    lines[i][j] = 0  # default to 0 if no previous score
        
        for i in range(num_islands):
            plt.plot(x, lines[i], label=f'Island {i}')
        
        plt.title('Island Performance Over Generations')
        plt.xlabel('Generations')
        plt.ylabel('Heuristic Score')
        plt.grid(True)
        plt.legend()
        plt.show()
        return  

vis = Vsualizations()
values = vis.get_best_entries()
print(values)