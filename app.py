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


app.layout = html.Div([
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        html.Div(className='container scalable', children=[
            html.H2(html.A(
                'Regression Explorer',
                href='https://github.com/plotly/dash-regression',
                style={'text-decoration': 'none', 'color': 'inherit'}
            )),
            html.A(
                html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                href='https://plot.ly/products/dash/'
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(
            className='row',
            style={'padding-bottom': '10px'},
            children=dcc.Markdown(dedent("""
            [Click here](https://github.com/plotly/dash-regression) to visit 
            the project repo, and learn about how to use the app.
            """))
        ),

        html.Div(id='custom-data-storage', style={'display': 'none'}),

        html.Div(className='row', children=[
            html.Div(className='four columns', children=drc.NamedDropdown(
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
            )),

            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Select Model',
                id='dropdown-select-model',
                options=[
                    {'label': 'Linear Regression', 'value': 'linear'},
                    {'label': 'Lasso', 'value': 'lasso'},
                    {'label': 'Ridge', 'value': 'ridge'},
                    {'label': 'Elastic Net', 'value': 'elastic_net'},
                ],
                value='linear',
                searchable=False,
                clearable=False
            )),

            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Click Mode (Select Custom Data to enable)',
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
            )),
        ]),

        html.Div(className='row', children=[
            html.Div(className='four columns', children=drc.NamedSlider(
                name='Polynomial Degree',
                id='slider-polynomial-degree',
                min=1,
                max=10,
                step=1,
                value=1,
                marks={i: i for i in range(1, 11)},
            )),

            html.Div(className='four columns', children=drc.NamedSlider(
                name='Alpha (Regularization Term)',
                id='slider-alpha',
                min=-4,
                max=3,
                value=0,
                marks={i: '{}'.format(10 ** i) for i in range(-4, 4)}
            )),

            html.Div(
                className='four columns',
                style={
                    'overflow-x': 'hidden',
                    'overflow-y': 'visible',
                    'padding-bottom': '10px'
                },
                children=drc.NamedSlider(
                    name='L1/L2 ratio (Select Elastic Net to enable)',
                    id='slider-l1-l2-ratio',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={0: 'L1', 1: 'L2'}
                )
            ),
        ]),

        dcc.Graph(
            id='graph-regression-display',
            className='row',
            style={'height': 'calc(100vh - 160px)'},
            config={'modeBarButtonsToRemove': [
                'pan2d',
                'lasso2d',
                'select2d',
                'autoScale2d',
                'hoverClosestCartesian',
                'hoverCompareCartesian',
                'toggleSpikelines'
            ]}
        ),
    ])
])


def make_dataset(name, random_state):
    np.random.seed(random_state)

    if name in ['sin', 'log', 'exp', 'tanh']:
        if name == 'sin':
            X = np.linspace(-np.pi, np.pi, 300)
            y = np.sin(X) + np.random.normal(0, 0.15, X.shape)
        elif name == 'log':
            X = np.linspace(0.1, 10, 300)
            y = np.log(X) + np.random.normal(0, 0.25, X.shape)
        elif name == 'exp':
            X = np.linspace(0, 3, 300)
            y = np.exp(X) + np.random.normal(0, 1, X.shape)
        elif name == 'tanh':
            X = np.linspace(-np.pi, np.pi, 300)
            y = np.tanh(X) + np.random.normal(0, 0.15, X.shape)
        return X.reshape(-1, 1), y

    elif name == 'boston':
        X = load_boston().data[:, -1].reshape(-1, 1)
        y = load_boston().target
        return X, y

    else:
        return make_regression(n_samples=300, n_features=1, noise=20,
                               random_state=random_state)


def format_coefs(coefs):
    coef_string = "yhat = "

    for order, coef in enumerate(coefs):
        if coef >= 0:
            sign = ' + '
        else:
            sign = ' - '
        if order == 0:
            coef_string += f'{coef}'
        elif order == 1:
            coef_string += sign + f'{abs(coef):.3f}*x'
        else:
            coef_string += sign + f'{abs(coef):.3f}*x^{order}'

    return coef_string


@app.callback(Output('slider-alpha', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(model):
    return model not in ['lasso', 'ridge', 'elastic_net']


@app.callback(Output('slider-l1-l2-ratio', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_dropdown_select_model(model):
    return model not in ['elastic_net']


@app.callback(Output('dropdown-custom-selection', 'disabled'),
              [Input('dropdown-dataset', 'value')])
def disable_custom_selection(dataset):
    return dataset != 'custom'

@app.callback(Output('custom-data-storage', 'children'),
              [Input('graph-regression-display', 'clickData')],
              [State('dropdown-custom-selection', 'value'),
               State('custom-data-storage', 'children'),
               State('dropdown-dataset', 'value')])
def update_custom_storage(clickData, selection, data, dataset):
    if data is None:
        data = {
            'train_X': [1, 2],
            'train_y': [1, 2],
            'test_X': [3, 4],
            'test_y': [3, 4],
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


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-alpha', 'value'),
               Input('dropdown-select-model', 'value'),
               Input('slider-l1-l2-ratio', 'value'),
               Input('custom-data-storage', 'children')])
def update_graph(dataset, degree, alpha_power, model_name, l2_ratio, custom_data):
    # Generate base data
    if dataset == 'custom':
        custom_data = json.loads(custom_data)
        X_train = np.array(custom_data['train_X']).reshape(-1, 1)
        y_train = np.array(custom_data['train_y'])
        X_test = np.array(custom_data['test_X']).reshape(-1, 1)
        y_test = np.array(custom_data['test_y'])
        X_range = np.linspace(-5, 5, 300).reshape(-1, 1)
        X = np.concatenate((X_train, X_test))

        trace_contour = go.Contour(
            x=np.linspace(-5, 5, 300),
            y=np.linspace(-5, 5, 300),
            z=np.ones(shape=(300, 300)),
            showscale=False,
            hoverinfo='none',
            contours=dict(coloring='lines'),
        )
    else:
        X, y = make_dataset(dataset, RANDOM_STATE)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=100, random_state=RANDOM_STATE)

        X_range = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # Create Polynomial Features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    poly_range = poly.fit_transform(X_range)

    # Select model
    alpha = 10 ** alpha_power
    if model_name == 'lasso':
        model = Lasso(alpha=alpha, normalize=True)
    elif model_name == 'ridge':
        model = Ridge(alpha=alpha, normalize=True)
    elif model_name == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=1 - l2_ratio, normalize=True)
    else:
        model = LinearRegression(normalize=True)

    # Train model and predict
    model.fit(X_train_poly, y_train)
    y_pred_range = model.predict(poly_range)
    test_score = model.score(X_test_poly, y_test)
    test_error = mean_squared_error(y_test, model.predict(X_test_poly))

    # Create figure
    trace0 = go.Scatter(
        x=X_train.squeeze(),
        y=y_train,
        name='Training Data',
        mode='markers',
        opacity=0.7,
    )
    trace1 = go.Scatter(
        x=X_test.squeeze(),
        y=y_test,
        name='Test Data',
        mode='markers',
        opacity=0.7,
    )
    trace2 = go.Scatter(
        x=X_range.squeeze(),
        y=y_pred_range,
        name='Prediction',
        mode='lines',
        hovertext=format_coefs(model.coef_)
    )
    data = [trace0, trace1, trace2]
    if dataset == 'custom':
        data.insert(0, trace_contour)

    layout = go.Layout(
        title=f"Score: {test_score:.3f}, MSE: {test_error:.3f} (Test Data)",
        legend=dict(orientation='h'),
        margin=dict(l=25, r=25),
        hovermode='closest'
    )

    return go.Figure(data=data, layout=layout)


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
