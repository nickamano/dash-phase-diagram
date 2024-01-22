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

    # Loop through each temperature
    for T in T_range:

        #################################
        # Free energies for pure A and pure B
        G_A = H_A - S_A * T
        G_B = H_B- S_B * T

        #################################
        # Free energies for dilute solution A
        G_Asolution = (1-x) * G_A + x * (G_B+1000) + R * T * (x * log(x) + (1-x) * log(1-x)) + x*(1-x)*wA
        G_Asolution[0] = G_A
        G_Asolution[len(x)-1] = G_B
        xG_Asolution = [(xx,yy,'A_solution') for xx, yy in zip(x, G_Asolution)]
        G_data['xG_Asolution'].append(xG_Asolution)

        # Free energies for dilute solution B
        G_Bsolution = (1-x) * (G_A+1000) + x * G_B + R * T * (x * log(x) + (1-x) * log(1-x)) + x*(1-x)*wB
        G_Bsolution[0] = G_A
        G_Bsolution[len(x)-1] = G_B
        xG_Bsolution = [(xx,yy,'B_solution') for xx, yy in zip(x, G_Bsolution)]
        G_data['xG_Bsolution'].append(xG_Bsolution)

        # Free energies for solid-solution AB
        G_ABsolution = (1-x) * G_A + x * G_B + R * T * (x * log(x) + (1-x) * log(1-x)) + x*(1-x)*w_AB
        G_ABsolution[0] = G_A
        G_ABsolution[len(x)-1] = G_B

        xG_ABsolution = xG_Bsolution = [(xx,yy,'AB_solution') for xx, yy in zip(x, G_ABsolution)]
        G_data['xG_ABsolution'].append(xG_ABsolution)

        #################################

        # Free energies for pure liquid A and B
        G_AL = H_AL - S_AL * T
        G_BL = H_BL - S_BL * T

        # Free energy of liquid
        G_liquid = (1-x) * G_AL + x * G_BL + R * T * (x * log(x) + (1-x) * log(1-x)) + x*(1-x)*(L0 + L1 * (x-(1-x)))
        G_liquid[0] = G_AL
        G_liquid[len(x)-1] = G_BL

        xG_liquid = [(xx,yy,'liquid') for xx, yy in zip(x, G_liquid)]
        G_data['xG_liquid'].append(xG_liquid)

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
            for every composition. Keys = phase_name. Values = list[list] where
            the inner list is over compositions and the outer list is over temperature.
        phases (list): The list of phases interested
        x (np.array): Composition vector

    Returns:
        figure: The plotly figure that contains the Gibbs free energy graphs of each phase and the convex hull of the 
    '''

    points_A = np.array([])
    points_A = np.array([])
    points_B = np.array([])
    points_AB = np.array([])
    points_liquid = np.array([])
    lower_hull = np.array([])
    x_3D = np.array([])

    # Remove phase labels
    for i in range(0, len(T_range)):
        point_A = [point[1] for point in G_data['xG_Asolution'][i]]
        point_B = [point[1] for point in G_data['xG_Bsolution'][i]]
        point_AB = [point[1] for point in G_data['xG_ABsolution'][i]]
        point_liquid = [point[1] for point in G_data['xG_liquid'][i]]
        points_A = np.append(points_A, point_A)
        points_B = np.append(points_B, point_B)
        points_AB = np.append(points_AB, point_AB)
        points_liquid = np.append(points_liquid, point_liquid)
        
        G_combine = np.concatenate((point_A, point_B, point_AB, point_liquid))
        points_G = np.stack((np.tile(x,4), G_combine)).T
        
        hull = ConvexHull(points_G)
        
        start = points_G[hull.vertices][points_G[hull.vertices][:, 0].argmin()]
        end = points_G[hull.vertices][points_G[hull.vertices][:, 0].argmax()]
        
        lower_hull_seg = np.array([point for point in [points_G[i] for i in hull.vertices] if below_line(point, start, end)])
        
        f = interpolate.interp1d(lower_hull_seg[:,0], lower_hull_seg[:,1],kind="linear")

        lower_hull = np.append(lower_hull, f(x)+5)
        x_3D = np.append(x_3D, x)
        
        

    # Generates 2D list of all temperature combinations for range of compositions
    T_3D = np.repeat(T_range, len(x))
    
    xx, yy = np.meshgrid(x, T_range)

    color_two_phase = 'gray'
    color_A = 'red'
    color_B = 'green'
    color_AB = 'yellow'
    color_liquid = 'blue'

    scatter_two_phase = go.Surface(x=xx, y=yy, z=lower_hull.reshape(len(T_range),len(x)), colorscale=[[0, color_two_phase], [1,color_two_phase]] , name='Two Phase', showscale=False)
    scatter_A = go.Surface(x=xx, y=yy, z=points_A.reshape(len(T_range),len(x)), colorscale=[[0, color_A], [1,color_A]], name='A', showscale=False)
    scatter_B = go.Surface(x=xx, y=yy, z=points_B.reshape(len(T_range),len(x)), colorscale=[[0, color_B], [1,color_B]], name='B', showscale=False)
    scatter_AB = go.Surface(x=xx, y=yy, z=points_AB.reshape(len(T_range),len(x)), colorscale=[[0, color_AB], [1,color_AB]], name='AB solution', showscale=False)
    scatter_liquid = go.Surface(x=xx, y=yy, z=points_liquid.reshape(len(T_range),len(x)), colorscale=[[0, color_liquid], [1,color_liquid]], name='liquid', showscale=False)

    if camera == None:
        camera = dict(up=dict(x=1, y=0, z=0),center=dict(x=0, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    
    layout = go.Layout(scene=dict(xaxis=dict(title='Composition [mol frac]', autorange="reversed"),
                              yaxis=dict(title='Temperature [K]',range=[300, 1500]),
                              zaxis=dict(title='Gibbs free energy [J/mol]'), camera = camera, dragmode="turntable"), 
                              margin=dict(l=0, r=0, t=10, b=10), height=630)
    fig = go.Figure(data=[scatter_two_phase, scatter_A, scatter_B, scatter_AB, scatter_liquid], layout=layout)
    return fig