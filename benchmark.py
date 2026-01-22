import time
import numpy as np
from convexhull import generate_data, visualize_convex_hull

def run_benchmark(nruns=10):
    phases = ['A_solution','B_solution', 'AB_solution', 'liquid']
    T_range = np.arange(1, 2000,5) # 400 steps
    x_grid = 0.01
    x = np.arange(0, 1+x_grid, x_grid)

    gen_times= 0
    vis_times = 0
    for i in range(nruns):
        print("Starting generation...")
        start = time.time()
        G_data = generate_data(x, T_range)
        gen_time = time.time() - start
        gen_times += gen_time
        print(f"Generate data: {gen_time:.4f}s")
    
        print("Starting visualization...")
        start = time.time()
        fig = visualize_convex_hull(G_data, phases, x, T_range)
        vis_time = time.time() - start
        vis_times += vis_time
        print(f"Visualize: {vis_time:.4f}s")
    
    print(f"Average Generate data: {gen_times/nruns:.4f}s")
    print(f"Average Visualize: {vis_times/nruns:.4f}s")
    print("Done.")

if __name__ == "__main__":
    run_benchmark()
