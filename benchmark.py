import time
import numpy as np
from convexhull import generate_data, visualize_convex_hull

def run_benchmark():
    phases = ['A_solution','B_solution', 'AB_solution', 'liquid']
    T_range = np.arange(1, 2000, 5) # 400 steps
    x_grid = 0.01
    x = np.arange(0, 1+x_grid, x_grid)

    print("Starting generation...")
    start = time.time()
    G_data = generate_data(x, T_range)
    gen_time = time.time() - start
    print(f"Generate data: {gen_time:.4f}s")
    
    print("Starting visualization...")
    start = time.time()
    # Mock camera to avoid None error if any (though code handles it)
    visualize_convex_hull(G_data, phases, x, T_range)
    vis_time = time.time() - start
    print(f"Visualize: {vis_time:.4f}s")

if __name__ == "__main__":
    run_benchmark()
