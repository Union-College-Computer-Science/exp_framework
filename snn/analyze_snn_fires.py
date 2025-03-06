"""
Analyze SNN firing patterns across generations from a CSV file. 

As John mentioned during class today, we want to find out how many times the SNN 
is firing during an evaluation run, which means 500 generations.

The way this works is by:
1) Load genomes from CSV files that contains the generations
2) Run simulations with genomes
3) Call run() function
4) We also added plotting relationships between firing patterns and fitness
5) What's the relationship between firing and performance?

Authors: Luodi Wang, Miguel Garduno, Atharv Tekurkar 
2/25/25
"""

import os 
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snn_sim.run_simulation import run

ITERS = 1000
GENOME_START_INDEX = 2

def analyze_fires(filename, generations=None, sample_count=5, mode="h"):
    """
    Analyze SNN firing patterns across generations.
    
    Parameters:
        filename (str): CSV file to analyze. Should be in /data directory.
        generations (list): List of specific generations to analyze. If None, will sample generations.
        sample_count (int): Number of generations to sample if generations is None.
        mode (str): Simulation mode ('h' for headless, 's' for screen, 'v' for video).
    """
    this_dir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(this_dir, os.path.join("data", filename)))
    
    max_gen = df['generation'].max()
    
    if generations is None:
        # Sample evenly spaced generations
        generations = np.linspace(1, max_gen, sample_count, dtype=int)
    
    print(f"\n===== Analyzing SNN Firing Patterns from {filename} =====")
    print(f"Looking at generations: {generations}")
    
    # Create a directory for results
    results_dir = os.path.join(this_dir, "analysis_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Store counted generations, fitness, and total firings of SNN
    all_gens = []
    all_fitness = []
    all_total_fires = []
    
    # Follow each generation
    for gen in generations:
        row = df.loc[df['generation'] == gen]
        if row.empty:
            print(f"Warning: Generation {gen} not found in CSV file. Skipping.")
            continue
            
        genome = row.values.tolist()[0][GENOME_START_INDEX:]
        fitness_from_csv = row.values.tolist()[0][1]  # Assuming fitness is in column 1
        
        print(f"\nProcessing Generation {gen} (Fitness from CSV: {fitness_from_csv})...")
        
        vid_name = f"{filename}_gen{gen}_analysis"
        vid_path = os.path.join(this_dir, "videos")
        
        # Run simulation: returns only fitness, not total fires
        fitness = run(ITERS, genome, mode, vid_name, vid_path)
        total_fires = 0  # Default value since run() doesn't return fire count
        
        print(f"Generation {gen} - Fitness: {fitness}, Total SNN Fires: {total_fires}")
        
        all_gens.append(gen)
        all_fitness.append(fitness)
        all_total_fires.append(total_fires)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(all_gens, all_fitness, 'b-o', label='Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness across Generations')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(all_gens, all_total_fires, 'r-o', label='Total SNN Fires')
    plt.xlabel('Generation')
    plt.ylabel('Total SNN Fires')
    plt.title('SNN Firing Activity across Generations')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{filename}_fire_analysis.png"))
    print(f"\nAnalysis complete! Results saved to {results_dir}")
    
    # Plot fires vs fitness
    plt.figure(figsize=(8, 6))
    plt.scatter(all_fitness, all_total_fires)
    plt.xlabel('Fitness')
    plt.ylabel('Total SNN Fires')
    plt.title('SNN Firing Activity vs Fitness')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"{filename}_fires_vs_fitness.png"))
    
    print(f"Total generations analyzed: {len(all_gens)}")
    if all_total_fires:
        print(f"Average fires per simulation: {sum(all_total_fires) / len(all_total_fires):.2f}")
        print(f"Maximum fires: {max(all_total_fires)} (Generation {all_gens[all_total_fires.index(max(all_total_fires))]})")
        print(f"Minimum fires: {min(all_total_fires)} (Generation {all_gens[all_total_fires.index(min(all_total_fires))]})")
        
        # Find correlation between fitness and firing
        if len(all_fitness) > 1:
            correlation = np.corrcoef(all_fitness, all_total_fires)[0, 1]
            print(f"Correlation between fitness and firing rate: {correlation:.4f}")
    
    # Show if mode is not headless
    if mode != "h":
        plt.show()

if __name__ == "__main__":
    csv_filename = "manualtesting.csv"
    
    analyze_fires(
        filename=csv_filename,
        generations=None,  # Set to a list like [1, 10, 50, 100] if you want specific generations
        sample_count=5,    # Number of generations to sample if generations is None
        mode="h"           # 'h' for headless, 's' for screen, 'v' for video
    )