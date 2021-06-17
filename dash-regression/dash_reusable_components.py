import dash_core_components as dcc
import dash_html_components as html


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def NamedSlider(name, **kwargs):
    return html.Div(
        style={'padding': '10px 10px 15px 4px', 'background-color':'rbg(17, 17, 102)', 'color':'rgb(237, 237, 247)'},
        children=[
            html.P(f'{name}:'),
            html.Div(dcc.Slider(**kwargs), style={'margin-left': '6px', 'background-color': '#3d1466'})
        ]
    )


def NamedDropdown(name, **kwargs):
    return html.Div([
        html.P(f'{name}:', style={'margin-left': '3px', 'background-color': 'rgb(35, 34, 69)', 'color': 'rgb(237, 237, 237)'}),
        dcc.Dropdown(**kwargs)
    ])

def banner(name, **kwargs):
    return html.Div(style={'padding': '10px 10px 15px 4px', 'background-color':'#3f1466'},)