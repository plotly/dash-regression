import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
server = app.server

RANDOM_STATE = 718

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


def make_sin():
    X = np.linspace(-np.pi, np.pi, 300)
    noise = np.random.normal(0, 0.1, X.shape)
    y = np.sin(X) + noise

    return X.reshape(-1, 1), y


def make_boston():
    boston = load_boston()
    X = boston.data[:, -1].reshape(-1, 1)
    y = boston.target

    return X, y

data_dict = {
    'linear': make_regression(n_samples=300, n_features=1, noise=20,
                              random_state=RANDOM_STATE),
    'sin': make_sin(),
    'boston': make_boston()
}


def serve_figure(n, dataset):
    # Generate figures
    X, y = data_dict[dataset]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=100,
                                                        random_state=RANDOM_STATE)

    X_train_poly = PolynomialFeatures(degree=n).fit_transform(X_train)
    X_test_poly = PolynomialFeatures(degree=n).fit_transform(X_test)

    X_range = np.arange(X.min() - 0.5, X.max() + 0.5, 0.2).reshape(-1, 1)
    poly_range = PolynomialFeatures(degree=n).fit_transform(X_range)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred_range = model.predict(poly_range)
    test_score = model.score(X_test_poly, y_test)

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
        mode='lines',
        name='Prediction'
    )
    data = [trace0, trace1, trace2]

    layout = go.Layout(
        title=f'Test Score: {test_score:.3f}',
        legend=dict(x=0, y=-0.01, orientation="h"),
    )

    return go.Figure(data=data, layout=layout)


app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        # Change App Name here
        html.Div(className='container scalable', children=[
            # Change App Name here
            html.H2('Regression Explorer'),

            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        # Add your code here

        html.Div(
            className='row',
            children=[
                html.Div(
                    className='four columns',
                    children=[
                        html.P('Polynomial Degree:'),

                        html.Div(
                            style={'margin-bottom': '50px'},
                            children=dcc.Slider(
                                id='slider-polynomial-degree',
                                min=1,
                                max=10,
                                step=1,
                                value=1,
                                marks={i: i for i in range(1, 11)},
                            )
                        ),

                        html.P('Dataset:'),

                        dcc.Dropdown(
                            id='dropdown-dataset',
                            options=[
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Sine Curve', 'value': 'sin'},
                                {'label': 'Boston', 'value': 'boston'}
                            ],
                            value='linear',
                            clearable=False,
                            searchable=False
                        )
                    ]
                ),

                html.Div(
                    className='eight columns',
                    style={'height': 'calc(100vh-85px)'},
                    children=dcc.Graph(id='graph-regression-display')
                )
            ]
        )
    ])
])


@app.callback(Output('graph-regression-display', 'figure'),
              [Input('slider-polynomial-degree', 'value'),
               Input('dropdown-dataset', 'value')])
def update_graph(degree, dataset):
    return serve_figure(degree, dataset)


external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet, replace this with your own base-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet, replace this with your own custom-styles.css using Rawgit
    "https://cdn.rawgit.com/xhlulu/638e683e245ea751bca62fd427e385ab/raw/fab9c525a4de5b2eea2a2b292943d455ade44edd/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
