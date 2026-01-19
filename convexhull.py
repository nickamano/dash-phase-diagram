from scipy.spatial import ConvexHull
from scipy import interpolate
import numpy as np
# Used to silence arithmetic errors that arise due to log
np.seterr(divide='ignore', invalid='ignore')
from numpy import log
import plotly.graph_objects as go

def generate_data(x, T_range, w_AB = 10000, L0= 5000, L1 =0, H_A = -35000, H_B = -38000, S_A = 11.1, S_B = 2.4, H_AL = -20000, H_BL = -25000, S_AL = 21.6, S_BL = 15, ):
    '''
    Builds the thermodynamic dataset of Gibbs Free energies using the regular
    solution model.

    Args:
        w_AB (float): The interaction parameter for the AB solution in J/mol
        L0 (float): The first (n = 0) Redlich-Kister parameter
        L1 (float): The second (n = 1) Redlich-Kister parameter

    Returns:
        G_data (dict): The Gibbs free energy (J/mol) for each phase at every temperature
            for every composition. Keys = phase_name. Values = list[list] where
            the inner list is over compositions and the outer list is over temperature.
    '''


    R = 8.314 #J/mol/K

    # The interaction parameter for dilute solution A and B (J/mol)
    wA = 15000
    wB = 15000

    # Initialize G_data dictionary
    G_data = {'xG_Asolution':[],'xG_Bsolution':[], 'xG_ABsolution':[], 'xG_liquid':[]}

    # Broadcast for vectorized interaction
    # T shape: (M,) -> (M, 1)
    # x shape: (N,) -> (1, N)
    T = T_range[:, np.newaxis]
    X = x[np.newaxis, :]
    
    ones_like_x = np.ones_like(X)
    
    # ----------------------------------------------
    # Calculate G for pure components and solutions
    # ----------------------------------------------
    
    # Pure A and B (M, 1)
    G_A = H_A - S_A * T
    G_B = H_B - S_B * T
    
    # Pre-calculate entropy term (M, N)
    # x*log(x) + (1-x)*log(1-x)
    # Handle x=0 and x=1 carefully to avoid NaN from log(0)
    # np.log(0) is -inf, 0*-inf is nan.
    # We use a safe computation
    with np.errstate(divide='ignore', invalid='ignore'):
        term_entropy = X * np.log(X) + (1 - X) * np.log(1 - X)
    term_entropy = np.nan_to_num(term_entropy) # Replace NaN with 0
    
    mix_entropy = R * T * term_entropy
    
    # --- Dilute Solution A ---
    # G = (1-x)GA + x(GB+1000) + RT(...) + x(1-x)wA
    # Shapes: (1,N)* (M,1) -> (M,N)
    G_Asolution = (1 - X) * G_A + X * (G_B + 1000) + mix_entropy + X * (1 - X) * wA
    # Boundaries: x=0 -> G_A, x=1 -> G_B
    # Note: X is (1,N). X[0,0] is 0. X[0,-1] is 1.
    # We enforce boundaries for all T
    G_Asolution[:, 0] = G_A[:, 0]
    G_Asolution[:, -1] = G_B[:, 0]
    G_data['xG_Asolution'] = G_Asolution

    # --- Dilute Solution B ---
    G_Bsolution = (1 - X) * (G_A + 1000) + X * G_B + mix_entropy + X * (1 - X) * wB
    G_Bsolution[:, 0] = G_A[:, 0]
    G_Bsolution[:, -1] = G_B[:, 0]
    G_data['xG_Bsolution'] = G_Bsolution

    # --- Solid Solution AB ---
    G_ABsolution = (1 - X) * G_A + X * G_B + mix_entropy + X * (1 - X) * w_AB
    G_ABsolution[:, 0] = G_A[:, 0]
    G_ABsolution[:, -1] = G_B[:, 0]
    G_data['xG_ABsolution'] = G_ABsolution

    # ----------------------------------------------
    # Liquid Phase
    # ----------------------------------------------
    G_AL = H_AL - S_AL * T
    G_BL = H_BL - S_BL * T
    
    G_liquid = (1 - X) * G_AL + X * G_BL + mix_entropy + X * (1 - X) * (L0 + L1 * (X - (1 - X)))
    G_liquid[:, 0] = G_AL[:, 0]
    G_liquid[:, -1] = G_BL[:, 0]
    G_data['xG_liquid'] = G_liquid

    return G_data

def below_line(point, start, end):
    '''
    Computes if a point is above or below a line connected by start and end point.

    Args:
        point (list): The x, y coordinates of the point of interest
        start (list): The first x, y coordinates of the start point
        end (list): The end x, y coordinates of the end point

    Returns:
        True if point is below line.
        False if point is above line
    '''
    return (point[1] - start[1]) * (end[0] - start[0]) <= (end[1] - start[1]) * (point[0] - start[0])


def visualize_convex_hull(G_data, phases, x, T_range, camera=None):
    '''
    Visualization tool in 3D to view the Gibbs free energy diagram.

    Args:
        G_data (dict): The Gibbs free energy (J/mol) for each phase at every temperature
            for every composition. Keys = phase_name. Values = np.array (M, N)
        phases (list): The list of phases interested
        x (np.array): Composition vector # (N,)

    Returns:
        figure: The plotly figure that contains the Gibbs free energy graphs of each phase and the convex hull of the 
    '''

    # Pre-allocate arrays for better performance
    n_temps = len(T_range)
    n_x = len(x)
    
    # G_data[...] are now (n_temps, n_x) arrays
    points_A = G_data['xG_Asolution']
    points_B = G_data['xG_Bsolution']
    points_AB = G_data['xG_ABsolution']
    points_liquid = G_data['xG_liquid']
    
    lower_hull = np.zeros((n_temps, n_x))

    # Remove phase labels
    for i in range(0, n_temps):
        # Extract data for this temperature step
        row_A = points_A[i, :]
        row_B = points_B[i, :]
        row_AB = points_AB[i, :]
        row_liquid = points_liquid[i, :]
        
        # Concatenate for ConvexHull
        G_combine = np.concatenate((row_A, row_B, row_AB, row_liquid))
        
        # Create (x, G) points. Tile x 4 times
        points_G = np.column_stack((np.tile(x, 4), G_combine))
        
        hull = ConvexHull(points_G)
        
        # Find start and end of the lower hull optimization
        hull_points = points_G[hull.vertices]
        start = hull_points[hull_points[:, 0].argmin()]
        end = hull_points[hull_points[:, 0].argmax()]
        
        # Filter for lower hull
        lower_hull_vertices = [point for point in hull_points if below_line(point, start, end)]
        lower_hull_seg = np.array(lower_hull_vertices)
        
        # Interpolate
        lower_hull_seg = lower_hull_seg[lower_hull_seg[:, 0].argsort()]
        
        f = interpolate.interp1d(lower_hull_seg[:,0], lower_hull_seg[:,1], kind="linear", bounds_error=False, fill_value="extrapolate")

        lower_hull[i, :] = f(x) + 5
        
        

    # Generates 2D list of all temperature combinations for range of compositions
    T_3D = np.repeat(T_range, len(x))
    
    xx, yy = np.meshgrid(x, T_range)

    color_two_phase = 'gray'
    color_A = 'red'
    color_B = 'green'
    color_AB = 'yellow'
    color_liquid = 'blue'

    scatter_two_phase = go.Surface(x=xx, y=yy, z=lower_hull, colorscale=[[0, color_two_phase], [1,color_two_phase]] , name='Two Phase', showscale=False)
    scatter_A = go.Surface(x=xx, y=yy, z=points_A, colorscale=[[0, color_A], [1,color_A]], name='A', showscale=False)
    scatter_B = go.Surface(x=xx, y=yy, z=points_B, colorscale=[[0, color_B], [1,color_B]], name='B', showscale=False)
    scatter_AB = go.Surface(x=xx, y=yy, z=points_AB, colorscale=[[0, color_AB], [1,color_AB]], name='AB solution', showscale=False)
    scatter_liquid = go.Surface(x=xx, y=yy, z=points_liquid, colorscale=[[0, color_liquid], [1,color_liquid]], name='liquid', showscale=False)

    if camera == None:
        camera = dict(up=dict(x=1, y=0, z=0),center=dict(x=0, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    
    layout = go.Layout(scene=dict(xaxis=dict(title='Composition [mol frac]', autorange="reversed"),
                              yaxis=dict(title='Temperature [K]',range=[300, 1500]),
                              zaxis=dict(title='Gibbs free energy [J/mol]'), camera = camera, dragmode="turntable"), 
                              margin=dict(l=0, r=0, t=10, b=10), height=630)
    fig = go.Figure(data=[scatter_two_phase, scatter_A, scatter_B, scatter_AB, scatter_liquid], layout=layout)
    return fig