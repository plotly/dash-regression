import os
import json
from textwrap import dedent

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from dash.dependencies import Input, Output, State

import dash_reusable_components as drc

RANDOM_STATE = 718

app = dash.Dash(__name__)
server = app.server

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

app.layout = html.Div(style={'width': '80%'}, children=[
    dcc.Graph(
        id='graph'
    ),

    drc.NamedDropdown(
        name='Select Dataset',
        id='dropdown-dataset',
        options=[
            {'label': 'Arctan Curve', 'value': 'tanh'},
            {'label': 'Boston (LSTAT Attribute)', 'value': 'boston'},
            {'label': 'Custom Data', 'value': 'custom'},
            {'label': 'Exponential Curve', 'value': 'exp'},
            {'label': 'Linear Curve', 'value': 'linear'},
            {'label': 'Log Curve', 'value': 'log'},
            {'label': 'Sine Curve', 'value': 'sin'},
        ],
        value='linear',
        clearable=False,
        searchable=False
    ),

    dcc.Dropdown(
        id='dropdown-custom-selection',
        options=[
            {'label': 'Add Training Data', 'value': 'training'},
            {'label': 'Add Test Data', 'value': 'test'},
            {'label': 'Remove Data point', 'value': 'remove'},
            {'label': 'Do Nothing', 'value': 'nothing'},
        ],
        value='training',
        clearable=False,
        searchable=False
    ),

    html.Div(id='json'),

    html.Div(id='custom-data-storage'),

    html.Button('Click me', id='button')
])


@app.callback(Output('custom-data-storage', 'children'),
              [Input('graph', 'clickData')],
              [State('dropdown-custom-selection', 'value'),
               State('custom-data-storage', 'children'),
               State('dropdown-dataset', 'value')])
def update_custom_storage(clickData, selection, data, dataset):
    if data is None:
        data = {
            'train_X': [],
            'train_y': [],
            'test_X': [],
            'test_y': [],
        }
    else:
        data = json.loads(data)
        if clickData and dataset == 'custom':
            selected_X = clickData['points'][0]['x']
            selected_y = clickData['points'][0]['y']

            if selection == 'training':
                data['train_X'].append(selected_X)
                data['train_y'].append(selected_y)
            elif selection == 'test':
                data['test_X'].append(selected_X)
                data['test_y'].append(selected_y)
            elif selection == 'remove':
                while selected_X in data['train_X'] and selected_y in data['train_y']:
                    data['train_X'].remove(selected_X)
                    data['train_y'].remove(selected_y)
                while selected_X in data['test_X'] and selected_y in data['test_y']:
                    data['test_X'].remove(selected_X)
                    data['test_y'].remove(selected_y)

    return json.dumps(data)


@app.callback(Output('graph', 'figure'),
              [Input('custom-data-storage', 'children')])
def func(custom_data_storage):
    data = json.loads(custom_data_storage)

    trace0 = go.Contour(
        x=np.linspace(0, 10, 200),
        y=np.linspace(0, 10, 200),
        z=np.ones(shape=(200, 200)),
        showscale=False,
        hoverinfo='none',
        contours=dict(coloring='lines'),
    )
    trace1 = go.Scatter(
        x=data['train_X'],
        y=data['train_y'],
        mode='markers',
        name='Training'
    )
    trace2 = go.Scatter(
        x=data['test_X'],
        y=data['test_y'],
        mode='markers',
        name='Training'
    )

    data = [trace0, trace1, trace2]
    figure = go.Figure(data=data)
    return figure


@app.callback(Output('json', 'children'),
              [Input('graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData)


external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet
    "https://cdn.rawgit.com/xhlulu/638e683e245ea751bca62fd427e385ab/raw/fab9c525a4de5b2eea2a2b292943d455ade44edd/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
