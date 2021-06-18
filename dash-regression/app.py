''' # IF U WANNA SOURCE THIS FILE:   
import os; os.system("source venv/bin/activate;")
'''
import os, json, re
from textwrap import dedent
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from sklearn.datasets import make_regression#, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
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

    html.Div(className='svg', style={'background-color': 'rgb(39, 229, 185)'}),
    html.Div(style={'background-color': 'rgb(39, 229, 185)'}),

    html.Div(className="banner", 
    style={'background-color': 'rgb(9, 0, 77)',},
    children=[
        html.Div(className='container scalable', children=[
            html.A(
                html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg", style={"height":"250px", "margin-top":"-35px", "margin-bottom":"-103px", "margin-right":"-100px"}),
                # html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg", style={"height":"250px", "margin-top":"-35px", "margin-bottom":"-100px", "margin-right":"-100px"}),
                # style={"background-color":'rgb(9, 0, 77)', 'margin-right':'100px'},
                # html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg", style={"height":"145px", "margin":"-60px -60px -10px -30px", "background-position":"center"}),
                # html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg", style={"height":"145px", "margin":"-20px", "background-position":"center"}),
                # html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg", style={"background-image":"url('https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg')", "width":"200px", "margin":"-20px", "background-position":"center"}),
                # style={"background-image":"url('https://i.stack.imgur.com/wPh0S.jpg')", "width":"200px", "height":"100px", "background-position":"center"},
                # html.Img(src="https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-7.jpg", style={'height':'145px'}),
                # html.Img(src="https://www.sji.edu.sg/qql/slot/u560/News%20and%20Events/News%20Highlights/2011/photos/arend%2019.jpg", style={'height':'120px'}),
                # html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                href='/'
                # href='https://plot.ly/products/dash/'
            ),
            # html.Span(
            #     "arend",
            #     style={"background-image":"url('https://www.redsports.sg/wp-content/uploads/2015/05/adiv-rugby-sf-acsi-ri-6.jpg')", "width":"40px", "height":"30px", "background-position":"center", 'font-size':'54pt'}
            # ),
            html.Span(html.A(
                "Arend's Riveting Regression Explorer",
                href='/',
                # href='https://github.com/plotly/dash-regression',
                className='col-4',
                style={'text-decoration': 'none', 'color': 'white', 'font-size':'25pt', 'font-weight':'bold',"background-color":'rgb(9, 0, 77)','height':'250px', 'padding-top':'100px', 'padding-bottom':'50px'}
            )),
        ]),
    ]),

    html.Div(id='body', className='container scalable col-10', 
        style={'background-color': 'rgb(9, 9, 21)',},
        children=[
            html.Div(
                className='row',
                # style={'padding-bottom': '10px', 'background-color': 'rgb(43, 9, 102)', 'color':'f5f5f5'},
                # children=dcc.Markdown(dedent("""
                # [Click here](https://github.com/plotly/dash-regression) to visit 
                # the project repo, and learn about how to use the app.
                # """))
        ),

        html.Div(id='custom-data-storage', style={'display': 'none'}),
        html.Div(id='react-entry-point', style={'background-color': 'rgb(39, 89, 185)'}),
        html.Div(className='svg-container', style={'background-color': 'rgb(39, 229, 185)'}),

        html.Div(style={'background-color': 'rgb(55, 8, 85)',},),

        html.Div(
            className='btn-group rounded px-3 pt-1 pb-2 col-12', 
            style={'background-color': 'rgb(35, 34, 69)',},
            children=[

                html.Div(
                    className='four columns col-4 py-0 my-0', 
                    style={'font-weight': 'bold',},
                    children=drc.NamedDropdown(
                        name='Select Dataset',
                        id='dropdown-dataset',
                        options=[
                            {'label': 'Billionaires Data', 'value': 'billionaires'},
                            {'label': 'Example Data: X.csv, y.csv', 'value': 'example_data'},
                            {'label': 'Custom Data', 'value': 'custom'},
                            {'label': 'Arctan Curve', 'value': 'tanh'},
                            # {'label': 'Boston (LSTAT Attribute)', 'value': 'boston'},
                            {'label': 'Exponential Curve', 'value': 'exp'},
                            {'label': 'Linear Curve', 'value': 'linear'},
                            {'label': 'Log Curve', 'value': 'log'},
                            {'label': 'Sine Curve', 'value': 'sin'},
                        ],
                        value='billionaires',
                        clearable=False,
                        searchable=False
                    )
                ),

                html.Div(
                    className='four columns col-4 py-0 my-0', 
                    style={'font-weight': 'bold',},
                    children=drc.NamedDropdown(
                        name='Select Model',
                        id='dropdown-select-model',
                        options=[
                            {'label': 'Linear Regression', 'value': 'linear'},
                            {'label': 'Logit', 'value': 'logit'},
                            {'label': 'Lasso', 'value': 'lasso'},
                            {'label': 'Ridge', 'value': 'ridge'},
                            {'label': 'Elastic Net', 'value': 'elastic_net'},
                        ],
                        value='linear',
                        searchable=False,
                        clearable=False
                )),

                html.Div(
                    className='four columns col-4 py-0 my-0', 
                    style={'font-weight': 'bold',},
                    children=drc.NamedDropdown(
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

         html.Div(
            id='_dash-app-content', 
            style={'background-color': 'rgb(40, 210, 13)',},
         ),

         html.Div(
            className='container.scalable', 
            style={'background-color': 'rgb(20, 25, 46)',},
         ),

         html.Div(
            className='container.scalable', 
            style={'background-color': 'rgb(19, 120, 43)',},
         ),

         html.Div(
            className='container', 
            style={'background-color': 'rgb(19, 12, 43)',},
         ),

         html.Div(
            className='scalable', 
            style={'background-color': 'rgb(19, 125, 43)',},
         ),

        #  html.Span(
        #      'Polynomial Degree',
        #      className='col-2 pb-0 pt-0 my-0', 
        #      style={'font-weight': 'bold'},
        #  ),

        html.Div(
            className='col-12 pb-3 pt-0 mt-0', 
            style={'background-color': 'rgb(9, 9, 26)', 'width':'100%'},
            children=[
                html.Div(className='six columns py-0 my-0 small', 
                style={'font-weight': 'bold'},
                children=drc.NamedSlider(
                    name='Polynomial Degree',
                    id='slider-polynomial-degree',
                    min=1,
                    max=10,
                    step=1,
                    value=1,
                    marks={i: i for i in range(1, 11)},
                )),

            html.Div(
                className='four columns', 
                style={'background-color': '#404040', 'display':'none',},
                children=drc.NamedSlider(
                    name='Alpha (Regularization Term)',
                    id='slider-alpha',
                    min=-4,
                    max=3,
                    value=0,
                    marks={i: '{}'.format(10 ** i) for i in range(-4, 4)}
                )
            ),

            html.Div(
                className='four columns',
                style={
                    'overflow-x': 'hidden',
                    'overflow-y': 'visible',
                    'padding-bottom': '10px', 'background-color': 'rgb(22, 28, 51)',
                    'display':'none',
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
            className='row py-0',
            style={'height': 'calc(100vh - 280px)', 'background-color': 'rgb(9, 9, 26)',},
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
], style={'background-color': 'rgb(9, 9, 21)',},)


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

    # elif name == 'boston':
    #     X = load_boston().data[:, -1].reshape(-1, 1)
    #     y = load_boston().target
    #     return X, y
    elif name == 'billionaires':
        # os.system('kaggle datasets download alexanderbader/forbes-billionaires-of-2021-20/forbes_billionaires.csv')
        # import zipfile 
        # archive = zipfile.ZipFile('forbes-billionaires-of-2021-20.zip')
        # filename = archive.filelist[0].filename
        # xlfile = archive.open(filename)
        # df = pd.read_csv(xlfile)
        df = pd.read_csv('https://raw.githubusercontent.com/srhoads/learn/main/data/forbes-billionaires-of-2021-20/forbes_billionaires.csv')
        Xvar = "Age"
        yvar = "NetWorth"
        Xy = df[[Xvar,yvar]].dropna().query(Xvar+'!=0 and '+yvar+'!=0 and '+Xvar+'!=1 and '+yvar+'!=1')
        # Xy = df[[Xvar,yvar]].dropna().apply(lambda c: c*1000 if 'NetWorth' in c.name else c).query(Xvar+'!=0 and '+yvar+'!=0 and '+Xvar+'!=1 and '+yvar+'!=1')
        # X = [[s] for s in Xy[Xvar].tolist()]
        X = Xy[Xvar].to_numpy().reshape(-1, 1)
        y = Xy[yvar].to_numpy()
        return X, y
        
    elif name == 'example_data':
        X = pd.read_csv('X.csv')
        y = pd.read_csv('y.csv')
        return X, y

    else:
        return make_regression(n_samples=300, n_features=1, noise=20, random_state=random_state)


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
            # bgcolor='#9d2929',
            showscale=False,
            hoverinfo='none',
            contours=dict(coloring='lines'),
        )
    else:
        # dataset.to_csv('dataset.csv', index=False)
        X, y = make_dataset(dataset, RANDOM_STATE) # dataset = pd.DataFrame(data)
        # pd.DataFrame(X).to_csv('X.csv', index=False); pd.DataFrame(y).to_csv('y.csv', index=False)
        # X = pd.read_csv('X.csv'); y = pd.read_csv('y.csv')

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=100, random_state=RANDOM_STATE)

        X_range = np.linspace(X.min() - 0.5, X.max() + 0.5, 300).reshape(-1, 1)

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # Create Polynomial Features
    poly = PolynomialFeatures(degree=degree) #degree=1
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
    if model_name == 'logit':
        # model = LogisticRegression()
        # X = X[2:-2]
        # y = np.array(y[2:-2]) # Getting rid of 0 and 1 values
        # y = np.log((1 / y) - 1)
        y_train = np.log(y_train)
        model = LinearRegression(normalize=False, fit_intercept=False)
    else:
        model = LinearRegression(normalize=True)

    # Train model and predict
    try:
        model.fit(X_train_poly, y_train)
    except:
        None
        # model.fit(X_train, y_train)
        # logit1 = sm.formula.logit(formula = "exh ~ hrs1 + age + prestg80 + babies", subset=(sub['wrkstat']==1), data = sub).fit()
        # logit1 = sm.formula.logit(formula = "exh ~ hrs1 + age + prestg80 + babies", subset=(sub['wrkstat']==1), data = sub).fit()
        # import statsmodels.discrete as smd
        # smd.discrete_model.Logit(X_train, y_train, check_rank=True)
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
        legend=dict(orientation='h', bgcolor="transparent",yanchor='top',y=1.035, font=dict(color='rgb(169, 172, 186)')),
        margin=dict(t=45, r=25, b=45, l=25),
        hovermode='closest',
        paper_bgcolor='rgb(49, 0, 0)',
        plot_bgcolor='rgb(66, 0, 0)',
        # bgcolor='#9d4342',
        # opacity=0.5,
    )

    # dict(
    #     paper_bgcolor="#F4F4F8",
    #     plot_bgcolor="#F4F4F8",
    #     autofill=True,
    #     margin=dict(t=75, r=50, b=100, l=50),
    # ),

    return go.Figure(data=data, layout=layout)


external_css = [
    # "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    # "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    # "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet
    # "https://cdn.rawgit.com/plotly/dash-regression/98b5a541/custom-styles.css",
    "https://cdn.jsdelivr.net/npm/startbootstrap-simple-sidebar@5.1.2/css/simple-sidebar.css",
    # "https://cdn.jsdelivr.net/npm/bs-custom-file-input/dist/bs-custom-file-input.js",
    "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    "https://cdn.datatables.net/buttons/1.6.4/css/buttons.dataTables.min.css",
    "https://cdn.datatables.net/1.10.22/css/jquery.dataTables.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
