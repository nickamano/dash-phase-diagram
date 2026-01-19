import numpy as np
from convexhull import generate_data, visualize_convex_hull

def debug_hover():
    phases = ['A_solution','B_solution', 'AB_solution', 'liquid']
    T_range = np.arange(1, 2000, 500) # Small range
    x_grid = 0.1
    x = np.arange(0, 1+x_grid, x_grid)

    print("Generating data...")
    G_data = generate_data(x, T_range)
    
    print("Visualizing...")
    fig = visualize_convex_hull(G_data, phases, x, T_range)
    
    # Check the Equilibrium trace (last one usually, or find by name)
    trace = None
    for t in fig.data:
        if t.name == 'Equilibrium':
            trace = t
            break
            
    if trace:
        print("Found Equilibrium trace.")
        if hasattr(trace, 'text'):
            print("Trace has 'text' attribute.")
            text_data = trace.text
            print(f"Text data type: {type(text_data)}")
            if isinstance(text_data, np.ndarray):
                print(f"Text data shape: {text_data.shape}")
                print(f"Sample text (middle): {text_data.flatten()[len(text_data.flatten())//2]}")
            else:
                print(f"Text data: {text_data}")
        else:
            print("Trace DOES NOT have 'text' attribute populated.")
            
        print(f"Hovertemplate: {trace.hovertemplate}")
    else:
        print("Equilibrium trace not found.")

if __name__ == "__main__":
    debug_hover()
