from dash import Dash, html, dcc, Input, Output, State, callback, clientside_callback
import numpy as np
from convexhull import generate_data, visualize_convex_hull

# Phases of interest
phases = ['A_solution','B_solution', 'AB_solution', 'liquid']
# Temperature range from 300 - 2000 (inclusive with 1K increments)
T_range = np.arange(1,2000,5)
# Mole fraction step size
x_grid = 0.01
# Mole fraction range from 0 to 1 (inclusive with x_grid increments)
x = np.arange(0,1+x_grid,x_grid)

# Used to silence arithmetic errors that arise due to log
np.seterr(divide='ignore', invalid='ignore')


app = Dash(__name__)

app.layout = html.Div([
                html.Div([
                    html.Div([
                        dcc.Markdown('''3D Gibbs Free Energy Graph using Regular Solutions Model '''),
                        dcc.Graph(id='3D-Gibbs', config=dict(responsive=True)),
                        html.Button('show phase diagram', id='camera-view', n_clicks=0, style={'margin':'2px'}),
                        html.Button('reset parameters', id='reset', n_clicks=0, style={'margin':'2px'})], style={'width':'60%', 'float': 'left', 'text-align':'center'}),
                    html.Div([
                        dcc.Markdown(id='w_AB-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=10000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='w-AB'),
                        dcc.Markdown(id='L0-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=5000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='L0'),
                        dcc.Markdown(id='L1-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=0,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='L1'),
                        dcc.Markdown(id='HA-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=-35000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='HA'),
                        dcc.Markdown(id='HB-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=-38000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='HB'),
                        dcc.Markdown(id='SA-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=11.1,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,10)},
                                id='SA'),
                        dcc.Markdown(id='SB-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=2.4,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,10)},
                                id='SB'),
                        dcc.Markdown(id='HAL-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=-20000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='HAL'),
                        dcc.Markdown(id='HBL-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                -100000, 100000, step=5,
                                value=-25000,
                                marks={-100000 + 20000*i : '{}'.format(-100000 + 20000*i) for i in range(0,10)},
                                id='HBL'),
                        dcc.Markdown(id='SAL-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=21.6,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,10)},
                                id='SAL'),
                        dcc.Markdown(id='SBL-disp', style={'marginTop': 20,}, mathjax=True), 
                        dcc.Slider(
                                0, 100, step=0.1,
                                value=15,
                                marks={i * 10 : '{}'.format(i*10) for i in range(0,10)},
                                id='SBL'),
                        ], style={'width': '39%', "text-align": "left", 'float':'left', 'min-width':'480px'})
                    ]),
            html.Footer([dcc.Markdown('''By Nicholas Amano at the University of Michigan''')])])

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
def update_omega_AB(layout, w_AB, L0, L1, HA, HB, SA, SB, HAL, HBL, SAL, SBL):
    if layout != None and 'scene.camera' in layout.keys():
        return visualize_convex_hull(generate_data(x, T_range, w_AB, L0, L1, HA, HB, SA, SB, HAL, HBL, SAL, SBL), phases, x, T_range, camera = layout['scene.camera'] )
    camera = dict(up=dict(x=1, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    return visualize_convex_hull(generate_data(x, T_range, w_AB, L0, L1, HA, HB, SA, SB, HAL, HBL, SAL, SBL), phases, x, T_range, camera = camera)

clientside_callback(
    "function(value) { return '$w_{ AB }$ Value: ' + value; }",
    Output('w_AB-disp', 'children'),
    Input('w-AB', 'value')
)

clientside_callback(
    "function(value) { return '$L_{ 0 }$ Value: ' + value; }",
    Output('L0-disp', 'children'),
    Input('L0', 'value')
)

clientside_callback(
    "function(value) { return '$L_{ 1 }$ Value: ' + value; }",
    Output('L1-disp', 'children'),
    Input('L1', 'value')
)

clientside_callback(
    "function(value) { return '$H_{ A }$ Value: ' + value; }",
    Output('HA-disp', 'children'),
    Input('HA', 'value')
)

clientside_callback(
    "function(value) { return '$H_{ B }$ Value: ' + value; }",
    Output('HB-disp', 'children'),
    Input('HB', 'value')
)

clientside_callback(
    "function(value) { return '$S_{ A }$ Value: ' + value; }",
    Output('SA-disp', 'children'),
    Input('SA', 'value')
)

clientside_callback(
    "function(value) { return '$S_{ B }$ Value: ' + value; }",
    Output('SB-disp', 'children'),
    Input('SB', 'value')
)

clientside_callback(
    "function(value) { return '$H_{ AL }$ Value: ' + value; }",
    Output('HAL-disp', 'children'),
    Input('HAL', 'value')
)

clientside_callback(
    "function(value) { return '$H_{ BL }$ Value: ' + value; }",
    Output('HBL-disp', 'children'),
    Input('HBL', 'value')
)

clientside_callback(
    "function(value) { return '$S_{ AL }$ Value: ' + value; }",
    Output('SAL-disp', 'children'),
    Input('SAL', 'value')
)

clientside_callback(
    "function(value) { return '$S_{ BL }$ Value: ' + value; }",
    Output('SBL-disp', 'children'),
    Input('SBL', 'value')
)

@callback(
    Output('3D-Gibbs', 'figure', allow_duplicate=True),
    Input('camera-view', 'n_clicks'),
    State('3D-Gibbs', 'figure'),
    prevent_initial_call=True)
def reset_view(_, figure):
    camera = dict(up=dict(x=1, y=0, z=0),eye=dict(x=0, y=0, z=-2.5))
    if 'layout' in figure.keys() and 'scene' in figure['layout'].keys() and figure['layout']['scene']['camera'] != camera:
        figure['layout']['scene']['camera'] = camera
    return figure

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
def reset_view(_):
    return visualize_convex_hull(generate_data(x, T_range), phases, x, T_range), 10000, 5000, 0, -35000, -38000, 11.1, 2.4, -20000, -25000, 21.6, 15

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)