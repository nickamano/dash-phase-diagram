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
            for every composition. Keys = phase_name. Values = np.ndarray (n_T, n_x)
    '''


    R = 8.314 #J/mol/K

    # The interaction parameter for dilute solution A and B (J/mol)
    wA = 15000
    wB = 15000

    # Create coordinate grids
    # xx, tt will have shape (len(T_range), len(x))
    xx, tt = np.meshgrid(x, T_range)

    #################################
    # Free energies for pure A and pure B (Temperature dependent)
    G_A = H_A - S_A * tt
    G_B = H_B - S_B * tt
    
    # Entropy of mixing term
    # x log x + (1-x) log (1-x)
    # Use np.nan_to_num to handle 0*log(0) = 0
    x_log_x = xx * np.log(xx)
    x_log_x = np.nan_to_num(x_log_x, nan=0.0)
    
    one_minus_x_log = (1-xx) * np.log(1-xx)
    one_minus_x_log = np.nan_to_num(one_minus_x_log, nan=0.0)
    
    S_mixing = R * tt * (x_log_x + one_minus_x_log)

    #################################
    # Free energies for dilute solution A
    G_Asolution = (1-xx) * G_A + xx * (G_B+1000) + S_mixing + xx*(1-xx)*wA
    # Enforce endpoints
    G_Asolution[:, 0] = G_A[:, 0]
    G_Asolution[:, -1] = G_B[:, 0]

    # Free energies for dilute solution B
    G_Bsolution = (1-xx) * (G_A+1000) + xx * G_B + S_mixing + xx*(1-xx)*wB
    G_Bsolution[:, 0] = G_A[:, 0]
    G_Bsolution[:, -1] = G_B[:, 0]

    # Free energies for solid-solution AB
    G_ABsolution = (1-xx) * G_A + xx * G_B + S_mixing + xx*(1-xx)*w_AB
    G_ABsolution[:, 0] = G_A[:, 0]
    G_ABsolution[:, -1] = G_B[:, 0]

    #################################

    # Free energies for pure liquid A and B
    G_AL = H_AL - S_AL * tt
    G_BL = H_BL - S_BL * tt

    # Free energy of liquid
    G_liquid = (1-xx) * G_AL + xx * G_BL + S_mixing + xx*(1-xx)*(L0 + L1 * (xx-(1-xx)))
    G_liquid[:, 0] = G_AL[:, 0]
    G_liquid[:, -1] = G_BL[:, 0]

    return {
        'xG_Asolution': G_Asolution,
        'xG_Bsolution': G_Bsolution,
        'xG_ABsolution': G_ABsolution,
        'xG_liquid': G_liquid
    }

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
            for every composition. Keys = phase_name. Values = list[list] where
            the inner list is over compositions and the outer list is over temperature.
        phases (list): The list of phases interested
        x (np.array): Composition vector
        T_range (np.array): Temperature vector

    Returns:
        figure: The plotly figure that contains the Gibbs free energy graphs of each phase and the convex hull
    '''

    # Pre-allocate lists for 3D surface data
    # We have len(T_range) rows, and each row has len(x) points
    n_T = len(T_range)
    n_x = len(x)
    
    # Extract data maps for easier access
    # G_data keys: 'xG_Asolution', 'xG_Bsolution', 'xG_ABsolution', 'xG_liquid'
    # Values are now np.ndarray of shape (n_T, n_x)

    G_A_all = G_data['xG_Asolution']
    G_B_all = G_data['xG_Bsolution']
    G_AB_all = G_data['xG_ABsolution']
    G_L_all = G_data['xG_liquid']
    
    # Store all G points for convex hull calculation
    # We handle this per temperature
    
    lower_hull_Z = np.zeros((n_T, n_x))
    lower_hull_Colors = np.zeros((n_T, n_x))
    
    # Phase Index Map for identifying vertices
    # 0: A, 1: B, 2: AB, 3: Liquid
    # Each block has n_x points.
    # Vertex index i:
    # 0 to n_x-1: A
    # n_x to 2n_x-1: B
    # 2n_x to 3n_x-1: AB
    # 3n_x to 4n_x-1: Liquid
    
    # Color codes for visualization
    # 0: Two-phase or Multi-phase (Default Gray) - We will override this
    # 1: A (Red)
    # 2: B (Green)
    # 3: AB (Yellow)
    # 4: Liquid (Blue)
    # 5: A + Liquid
    # 6: B + Liquid
    # 7: AB + Liquid
    # 8: A + B
    # 9: A + AB
    # 10: B + AB
    
    # Mapping pairs to color codes
    # Key: frozenset({phase_idx1, phase_idx2})
    # pure phases are handled if p1==p2
    
    phase_indices = {0: 'A', 1: 'B', 2: 'AB', 3: 'L'}
    
    def get_color_code(p1, p2):
        s = frozenset([p1, p2])
        if len(s) == 1:
            p = list(s)[0]
            if p == 0: return 1 # A
            if p == 1: return 2 # B
            if p == 2: return 3 # AB
            if p == 3: return 4 # L
        else:
            if s == frozenset([0, 3]): return 5 # A + L
            if s == frozenset([1, 3]): return 6 # B + L
            if s == frozenset([2, 3]): return 7 # AB + L
            if s == frozenset([0, 1]): return 8 # A + B
            if s == frozenset([0, 2]): return 9 # A + AB
            if s == frozenset([1, 2]): return 10 # B + AB
        return 0 # Fallback

    for i in range(n_T):
        # Gather points for this temp
        # x is constant
        g_A = G_A_all[i]
        g_B = G_B_all[i]
        g_AB = G_AB_all[i]
        g_L = G_L_all[i]
        
        # Concatenate G values
        G_combine = np.concatenate((g_A, g_B, g_AB, g_L))
        # Create (x, G) points. We repeat x 4 times.
        x_repeated = np.tile(x, 4)
        points_G = np.column_stack((x_repeated, G_combine))
        
        hull = ConvexHull(points_G)
        
        # Identify the lower hull
        start_idx = hull.vertices[points_G[hull.vertices, 0].argmin()]
        end_idx = hull.vertices[points_G[hull.vertices, 0].argmax()]
        
        start_pt = points_G[start_idx]
        end_pt = points_G[end_idx]
        
        # Filter vertices that are part of the lower hull (below the line connecting min_x and max_x)
        # Note: 'below_line' check is good, but for strictly convex lower hull, 
        # we can also just check which simplices (edges in 2D) have normal pointing down.
        # But continuing with existing logic of vertex filtering + interpolation.
        
        # Get vertices on lower hull
        # Optimization: We can just use the simplices that form the lower boundary.
        # SciPy 2D Hull simplices are line segments [i, j].
        # We want the bottom chain.
        # Simpler approach matching previous logic but optimized:
        
        vertices_lower = [v for v in hull.vertices if below_line(points_G[v], start_pt, end_pt)]
        # Add start and end to ensure they are included (sometimes rounding issues?)
        vertices_lower = sorted(list(set(vertices_lower + [start_idx, end_idx])), key=lambda v: points_G[v, 0])
        
        lower_points = points_G[vertices_lower]
        lower_indices = vertices_lower
        
        # Now we interpolate across x
        # usage: interpolate.interp1d
        # But we also need to interpolate COLOR.
        # This means for every x, we need to know which segment [v_left, v_right] it falls on.
        
        # Since lower_points are sorted by x, we can use searchsorted or just iterate.
        # Given n_x is small (~100), iteration or simple lookup is fine. 
        # But interp1d is fast for Z values.
        
        f = interpolate.interp1d(lower_points[:,0], lower_points[:,1], kind="linear", fill_value="extrapolate")
        lower_hull_Z[i, :] = f(x)
        
        # For coloring, we determine the segment for each x
        # x values are sorted. lower_points are sorted by x.
        # We can find where each x falls in lower_points x-coordinates.
        
        # Indices of bins to which each value in x belongs
        # bins[j] = i means lower_points[i-1] <= x[j] < lower_points[i]
        bins = np.searchsorted(lower_points[:, 0], x)
        
        # Clip bins to valid range for segments
        # Valid segments are 0 to len(lower_points)-2
        # If bin is 0, it means x < first_point (shouldn't happen if x covers range, but safety)
        # If bin is len, x >= last_point
        
        # We want segment index k such that lower_points[k] <= x <= lower_points[k+1]
        # searchsorted returns insertion point.
        # value in [k, k+1] -> bin k+1. So segment is bin-1.
        
        seg_indices = bins - 1
        seg_indices = np.clip(seg_indices, 0, len(lower_points)-2)
        
        # Now map segment index to color code
        # A segment connects vertices_lower[k] and vertices_lower[k+1]
        # We need the original indices to determine phase.
        
        current_colors = []
        # Precompute colors for each segment to avoid re-evaluating inside loop
        seg_color_map = []
        for k in range(len(lower_points)-1):
            idx1 = lower_indices[k]
            idx2 = lower_indices[k+1]
            
            # Phase = idx // n_x
            p1 = idx1 // n_x
            p2 = idx2 // n_x
            
            code = get_color_code(p1, p2)
            seg_color_map.append(code)
            
        seg_color_map = np.array(seg_color_map)
        
        # Assign colors
        lower_hull_Colors[i, :] = seg_color_map[seg_indices]


    # Create grid for plotting
    xx, yy = np.meshgrid(x, T_range)
    
    # Custom colorscale for our discrete phase codes
    # 1: A (Red), 2: B (Green), 3: AB (Yellow), 4: L (Blue)
    # 5: A+L (Purple), 6: B+L (Cyan), 7: AB+L (Orange), 8: A+B (Grey), 9: A+AB (Pink?), 10: B+AB (Lime?)
    
    # Let's define a discrete map. Plotly surfaces use continuous numbers mapped to colors.
    # We can handle this by defining tickvals and exact colors.
    
    # Colors string definitions
    c_A = 'red'
    c_B = 'green'
    c_AB = 'yellow'
    c_L = 'blue'
    c_AL = 'purple'
    c_BL = 'cyan'
    c_ABL = 'orange'
    c_AB_mix = 'brown' # A+B or others, unlikely for this simple eutectic but possible
    
    # We need to constructing a colorscale that maps specific integers to specific colors.
    # range 0 to 10.
    
    discrete_colors = [
        [0.0, 'gray'], [0.09, 'gray'],
        [0.1, 'red'], [0.19, 'red'],       # 1
        [0.2, 'green'], [0.29, 'green'],   # 2
        [0.3, 'yellow'], [0.39, 'yellow'], # 3
        [0.4, 'blue'], [0.49, 'blue'],     # 4
        [0.5, 'purple'], [0.59, 'purple'], # 5
        [0.6, 'cyan'], [0.69, 'cyan'],     # 6
        [0.7, 'orange'], [0.79, 'orange'], # 7
        [0.8, 'brown'], [1.0, 'brown']     # 8+
    ]

    # Map color codes to text descriptions for hover
    # 0: Fallback, 1: A, 2: B, 3: AB, 4: L
    # 5: A+L, 6: B+L, 7: AB+L, 8: Solid Mix
    
    code_to_name = {
        0: 'Unknown', 1: 'Phase A', 2: 'Phase B', 3: 'Phase AB', 4: 'Liquid',
        5: 'A + Liquid', 6: 'B + Liquid', 7: 'AB + Liquid', 8: 'Solid Mixture (A+B)',
        9: 'Solid Mixture (A+AB)', 10: 'Solid Mixture (B+AB)'
    }
    
    # Vectorize the mapping
    # Since keys are small integers, we can use a lookup array or just list comprehension
    # lower_hull_Colors is float (from surfacecolor), cast to int for lookup
    
    color_ints = lower_hull_Colors.astype(int)
    
    # Quick lookup function
    def lookup_name(code):
        return code_to_name.get(code, 'Two Phase')
        
    # Apply lookup to create text matrix
    # np.vectorize might be slowish but fine for 400x100
    v_lookup = np.vectorize(lookup_name)
    hover_texts = v_lookup(color_ints).tolist()

    scatter_two_phase = go.Surface(
        x=xx, y=yy, z=lower_hull_Z, 
        surfacecolor=lower_hull_Colors,
        cmin=0, cmax=10,
        colorscale=discrete_colors,
        name='Equilibrium',
        showscale=False,
        text=hover_texts,
        hovertemplate="<b>%{text}</b><br>Composition: %{x:.2f}<br>Temperature: %{y:.0f} K<br>Gibbs Energy: %{z:.0f} J/mol<extra></extra>"
    )

    scatter_A = go.Surface(x=xx, y=yy, z=G_A_all, colorscale=[[0, c_A], [1,c_A]], name='A', showscale=False, opacity=0.3,
                           hovertemplate="<b>Phase A</b><br>Composition: %{x:.2f}<br>Temperature: %{y:.0f} K<br>Gibbs Energy: %{z:.0f} J/mol<extra></extra>")
    scatter_B = go.Surface(x=xx, y=yy, z=G_B_all, colorscale=[[0, c_B], [1,c_B]], name='B', showscale=False, opacity=0.3,
                           hovertemplate="<b>Phase B</b><br>Composition: %{x:.2f}<br>Temperature: %{y:.0f} K<br>Gibbs Energy: %{z:.0f} J/mol<extra></extra>")
    scatter_AB = go.Surface(x=xx, y=yy, z=G_AB_all, colorscale=[[0, c_AB], [1,c_AB]], name='AB solution', showscale=False, opacity=0.3,
                            hovertemplate="<b>Phase AB</b><br>Composition: %{x:.2f}<br>Temperature: %{y:.0f} K<br>Gibbs Energy: %{z:.0f} J/mol<extra></extra>")
    scatter_liquid = go.Surface(x=xx, y=yy, z=G_L_all, colorscale=[[0, c_L], [1,c_L]], name='liquid', showscale=False, opacity=0.3,
                                hovertemplate="<b>Liquid Phase</b><br>Composition: %{x:.2f}<br>Temperature: %{y:.0f} K<br>Gibbs Energy: %{z:.0f} J/mol<extra></extra>")

    if camera == None:
        camera = dict(up=dict(x=1, y=0, z=0),center=dict(x=0, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    
    layout = go.Layout(scene=dict(xaxis=dict(title='Composition [mol frac]', autorange="reversed"),
                              yaxis=dict(title='Temperature [K]',range=[300, 1500]),
                              zaxis=dict(title='Gibbs free energy [J/mol]'), camera = camera, dragmode="turntable"), 
                              margin=dict(l=0, r=0, t=10, b=10), height=630)
    
    # User requested full curves to be present. 
    # Adding scatter_two_phase Last so it renders on top? or First? 
    # In 3D opaque surfaces occlude each other.
    # I set opacity=0.3 for the constituent phases so the hull (Equilibrium) is visible inside/underneath.
    # Actually, the lower hull is the *minimum*, so it will be "under" the others visually if looking from top, 
    # but "covering" them if looking from bottom.
    # The user wants "viewed from the bottom, the 3D graphic will represent a 2D phase diagram".
    # This implies the lower hull should be opaque and identifiable.
    
    fig = go.Figure(data=[scatter_A, scatter_B, scatter_AB, scatter_liquid, scatter_two_phase], layout=layout)
    return fig