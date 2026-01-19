from dash import Dash, html, dcc, Input, Output, State, callback, Patch
import numpy as np
import flask
from app.app.convexhull import generate_data, visualize_convex_hull

# Phases of interest
phases = ['A_solution','B_solution', 'AB_solution', 'liquid']
# Temperature range from 300 - 2000 (inclusive with 1K increments)
T_range = np.arange(300,2000,6)
# Mole fraction step size
x_grid = 0.02
# Mole fraction range from 0 to 1 (inclusive with x_grid increments)
x = np.arange(0,1+x_grid,x_grid)

# Used to silence arithmetic errors that arise due to log
np.seterr(divide='ignore', invalid='ignore')

footer = '''Made By [Nicholas Amano](https://websites.umich.edu/~namano/) at the University of Michigan.

The above is a visualization demonstrating how phase diagrams are made. Using the regular solution model of the entropy of mixing, 
we derive the Gibbs free energy curves of a solution A, a solution B, a liquid phase, and an AB solution phase in 
composition and temperature space. We then observe if the convex hull of the curves lies on any given solution, if it does, then 
that is the energetically stable phase, and if not, a two phase region is stable. Here, two phase regions is shaded __gray__,
the pure single phases are __green__ and __red__, the liquid phase is __blue__, and the mixed solution is __yellow__. 
'''

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = flask.Flask(__name__)
dash_app = Dash(__name__, external_stylesheets=external_stylesheets, server = app, requests_pathname_prefix = '/rsm/')
fig = visualize_convex_hull(generate_data(x, T_range), phases, x, T_range)

dash_app.layout = html.Div([
                html.Div([dcc.Markdown('''## 3D Gibbs Free Energy for Regular Solutions Model''', style={'text-align': 'center'}), 
                    html.Div([
                        dcc.Graph(figure = fig, id='3D-Gibbs', config=dict(responsive=True)),
                        html.Button('reset parameters', id='reset', n_clicks=0, style={'margin':'2px'})], style={'width':'60%', 'float': 'left', 'text-align':'center'}),
                    html.Div([
                        dcc.Markdown(id='w_AB-disp', style={'marginTop': 20}, mathjax=True),    
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=20,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},
                                id='w-AB'),
                        dcc.Markdown(id='L0-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -30, 20, step=0.1,
                                value=0,
                                marks={5* i - 30 : '{}'.format(5*i - 30) for i in range(0,11)},
                                id='L0'),
                        dcc.Markdown(id='L1-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=10,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},
                                id='L1'),
                        dcc.Markdown(id='HA-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=-35,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},
                                id='HA'),
                        dcc.Markdown(id='HB-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=-36,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},\
                                id='HB'),
                        dcc.Markdown(id='SA-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=11.1,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,11)},
                                id='SA'),
                        dcc.Markdown(id='SB-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=2.4,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,11)},
                                id='SB'),
                        dcc.Markdown(id='HAL-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=-20,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},
                                id='HAL'),
                        dcc.Markdown(id='HBL-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                -100, 100, step=.1,
                                value=-25,
                                marks={-100 + 20*i : '{}'.format(-100 + 20*i) for i in range(0,11)},
                                id='HBL'),
                        dcc.Markdown(id='SAL-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=21.6,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,11)},
                                id='SAL'),
                        dcc.Markdown(id='SBL-disp', style={'marginTop': 20}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=15,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,11)},
                                id='SBL'),
                        ], style={'width': '39%', "text-align": "left", 'float':'left', 'min-width':'480px'})
                    ]), html.Div([dcc.Markdown(footer)], style={'float':'left', 'clear': 'left'})])

@callback(
    Output('3D-Gibbs', 'figure'),
    State('3D-Gibbs', 'relayoutData'),
    Input('w-AB', 'value'),
    Input('L0', 'value'),
    Input('L1', 'value'),
    Input('HA', 'value'),
    Input('HB', 'value'),
    Input('SA', 'value'),
    Input('SB', 'value'),
    Input('HAL', 'value'),
    Input('HBL', 'value'),
    Input('SAL', 'value'),
    Input('SBL', 'value'))
def update_params(layout, w_AB, L0, L1, HA, HB, SA, SB, HAL, HBL, SAL, SBL):
    if layout != None and 'scene.camera' in layout.keys():
        return visualize_convex_hull(generate_data(x, T_range, w_AB * 1000, L0 * 1000, L1 * 1000, HA* 1000, HB* 1000, SA, SB, HAL* 1000, HBL* 1000, SAL, SBL), phases, x, T_range, camera = layout['scene.camera'] )
    camera = dict(up=dict(x=1, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    patch_figure = Patch()
    patch_figure['data'] = visualize_convex_hull(generate_data(x, T_range, w_AB * 1000, L0 * 1000, L1 * 1000, HA* 1000, HB* 1000, SA, SB, HAL* 1000, HBL* 1000, SAL, SBL), phases, x, T_range, camera = camera).data
    return patch_figure

@callback(
    Output('w_AB-disp', 'children'),
    Input('w-AB', 'value'))
def display_value(value):
    return fr'$w_{{ AB }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('L0-disp', 'children'),
    Input('L0', 'value'))
def display_value(value):
    return fr'$L_{{ 0 }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('L1-disp', 'children'),
    Input('L1', 'value'))
def display_value(value):
    return fr'$L_{{ 1 }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('HA-disp', 'children'),
    Input('HA', 'value'))
def display_value(value):
    return fr'$H_{{ A }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('HB-disp', 'children'),
    Input('HB', 'value'))
def display_value(value):
    return fr'$H_{{ B }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('SA-disp', 'children'),
    Input('SA', 'value'))
def display_value(value):
    return fr'$S_{{ A }}$: ${value} \frac{{kJ}}{{mol \times K}}$'

@callback(
    Output('SB-disp', 'children'),
    Input('SB', 'value'))
def display_value(value):
    return fr'$S_{{ B }}$: ${value} \frac{{kJ}}{{mol \times K}}$'

@callback(
    Output('HAL-disp', 'children'),
    Input('HAL', 'value'))
def display_value(value):
    return fr'$H_{{ AL }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('HBL-disp', 'children'),
    Input('HBL', 'value'))
def display_value(value):
    return fr'$H_{{ BL }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('SAL-disp', 'children'),
    Input('SAL', 'value'))
def display_value(value):
    return fr'$S_{{ AL }}$: ${value} \frac{{kJ}}{{mol}}$'

@callback(
    Output('SBL-disp', 'children'),
    Input('SBL', 'value'))
def display_value(value):
    return fr'$S_{{ BL }}$: ${value} \frac{{kJ}}{{mol}}$'


@callback(
    [Output('3D-Gibbs', 'figure', allow_duplicate=True),
    Output('w-AB', 'value'),
    Output('L0', 'value'),
    Output('L1', 'value'),
    Output('HA', 'value'),
    Output('HB', 'value'),
    Output('SA', 'value'),
    Output('SB', 'value'),
    Output('HAL', 'value'),
    Output('HBL', 'value'),
    Output('SAL', 'value'),
    Output('SBL', 'value')],
    Input('reset', 'n_clicks'),
    prevent_initial_call=True)
def reset_values(_):
    return visualize_convex_hull(generate_data(x, T_range), phases, x, T_range), 10, 0, 10, -35000, -38000, 11.1, 2.4, -20000, -25000, 21.6, 15

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)