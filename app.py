import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from plotly import tools
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from dash.dependencies import Input, Output, State

import dash_reusable_components as drc

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
            html.H2('Regression Explorer'),

            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(className='row', children=[
            html.Div(className='four columns', children=drc.NamedDropdown(
                name='Select Dataset',
                id='dropdown-dataset',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Sine Curve', 'value': 'sin'},
                    {'label': 'Boston', 'value': 'boston'}
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

            html.Div(className='four columns', children=drc.NamedSlider(
                name='Polynomial Degree',
                id='slider-polynomial-degree',
                min=1,
                max=10,
                step=1,
                value=1,
                marks={i: i for i in range(1, 11)},
            )),
        ]),

        html.Div(className='row', children=[
            html.Div(className='six columns', children=drc.NamedSlider(
                name='Alpha (Regularization Term)',
                id='slider-alpha',
                min=-4,
                max=3,
                value=0,
                marks={i: '{}'.format(10 ** i) for i in range(-4, 4)}
            )),

            html.Div(
                className='six columns',
                style={
                    'overflow-x': 'hidden',
                    'overflow-y': 'visible',
                    'padding-bottom': '30px'
                },
                children=drc.NamedSlider(
                    name='L1/L2 ratio',
                    id='slider-l1-l2-ratio',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={0: 'L1', 1: 'L2'}
                )),
        ]),

        html.Div(
            children=dcc.Graph(
                id='graph-regression-display',
                className='row',
                style={'height': 'calc(100vh - 205px)'},
            )
        )
    ])
])


def make_dataset(name):
    if name == 'sin':
        X = np.linspace(-np.pi, np.pi, 300)
        noise = np.random.normal(0, 0.1, X.shape)
        y = np.sin(X) + noise
        return X.reshape(-1, 1), y

    elif name == 'boston':
        boston = load_boston()
        X = boston.data[:, -1].reshape(-1, 1)
        y = boston.target
        return X, y

    else:
        return make_regression(n_samples=300, n_features=1, noise=20,
                               random_state=718)


@app.callback(Output('slider-alpha', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(dataset):
    return dataset not in ['lasso', 'ridge', 'elastic_net']


@app.callback(Output('slider-l1-l2-ratio', 'disabled'),
              [Input('dropdown-select-model', 'value')])
def disable_slider_alpha(dataset):
    return dataset not in ['elastic_net']


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('dropdown-dataset', 'value'),
               Input('slider-polynomial-degree', 'value'),
               Input('slider-alpha', 'value'),
               Input('dropdown-select-model', 'value'),
               Input('slider-l1-l2-ratio', 'value')])
def update_graph(dataset, degree, alpha_power, model_name, l2_ratio):
    random_state = 718
    alpha = 10 ** alpha_power

    # Generate base data
    X, y = make_dataset(dataset)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=100, random_state=random_state)
    X_range = np.arange(X.min() - 0.5, X.max() + 0.5, 0.2).reshape(-1, 1)

    # Create Polynomial Features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    poly_range = poly.fit_transform(X_range)

    # Select model
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
    coefs = model.coef_[1:]

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
    )

    trace3 = go.Bar(
        x=[f'x^{p}' for p in range(1, len(coefs) + 1)],
        y=coefs,
        xaxis='x2',
        yaxis='y2',
        name='Coefficients'
    )

    data = [trace0, trace1, trace2, trace3]
    layout = go.Layout(
        title=f"Score: {test_score:.3f}, MSE: {test_error:.3f} (Test Data)",
        xaxis=dict(domain=[0, 0.65]),
        xaxis2=dict(domain=[0.7, 1]),
        yaxis2=dict(anchor='x2'),
        legend=dict(orientation='h'),
        margin=dict(l=25, r=25)
    )

    return go.Figure(data=data, layout=layout)


external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
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
