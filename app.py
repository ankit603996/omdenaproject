### Import Packages ########################################
import os
import plotly.express as px
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pickle
import shap
import dill
from chart_studio import plotly
import os
import io,base64

import requests

def create_tf_serving_json(data):
    return [{'inputs': {name: data[name] for name in data.keys()} if isinstance(data, dict) else data.tolist()}['inputs']]


def score_model(data):
    url = 'https://adb-1892098852338246.6.azuredatabricks.net/model/Omdena2/2/invocations'
    headers = {'Authorization': f'Bearer dapicd59a55031a134e2c1787e88043f540d'}
    data = {"dataframe_split": data.to_dict(orient='split')}
    response = requests.request(method='POST', headers=headers, url=url, json=data)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

### Setup ###################################################
dash_app = dash.Dash(__name__,prevent_initial_callbacks=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app.title = 'Machine Learning Model Deployment'
app = dash_app.server

### App Layout ###############################################
dash_app.layout = html.Div([
    html.H1("MobileUurka Risk Prediction Model", style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    dcc.Tabs(id="tabs-example-graph",  children=[
        dcc.Tab(label='Batch Prediction', value='tab-1-example-graph'),
        dcc.Tab(label='Single Prediction', value='tab-2-example-graph'),
    ]),
    html.Div(id='tabs-content-example-graph')
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')))
    return df

###UI
@dash_app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-example-graph':
        return   html.Div([
                    dbc.Row([html.H3(children='''Upload Patient's Records ''')]),
                    html.Br(),
                    html.Br(),
                    dbc.Row([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    )]),
                dbc.Row([
                    html.Div(id='output-data-upload')
                ]),
                html.Br(),
                html.Br(),
                dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary"),
                dbc.Row([html.Div(id='prediction output')]),
                html.Br(),
                ])
    elif tab == 'tab-2-example-graph':
         return   html.Div([
                dbc.Row([html.H3(children='Provide Patient Information')]),
                html.Br(),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(html.Label(children='Age (years):')),
                            dbc.Col(dcc.Input(type="number", min=0, value=25, id='Age'))
                        ]),
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(html.Label(children='SystolicBP')),
                            dbc.Col(dcc.Input(type="number", min=0, value=90, id='SystolicBP')
                                    )
                        ]),
                        dbc.Row(html.Br()),
                        ],width="auto"),
                    dbc.Col([
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(html.Label(children='DiastolicBP of patient')),
                            dbc.Col(dcc.Input(type="number", min=0, value=90, id='DiastolicBP'))
                        ]),
                        dbc.Row(html.Br()),
                        dbc.Row([
                            dbc.Col(html.Label(children='Blood sugar of patient')),
                            dbc.Col(dcc.Input(type="number", min=0, value=90, id='BS')),
                        dbc.Row(html.Br()),
                        ])
                        ],width="auto")
            ],justify="start",style={"border":"2px black solid","background":"grey"},
                align="center"),
        html.Br(),
        dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary"),
        html.Br(),
        html.Br(),
        dbc.Row([html.Div(id='prediction output2')])
    ])


### Callback to produce the prediction #########################
@dash_app.callback(
    Output('prediction output', 'children'),
    Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified')
)

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df_list = [parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        df = df_list[0]
        data = df.copy()
        prediction_batch = pd.DataFrame(score_model(data))
        prediction_batch.columns = ['ML_Prediction']
        conditions = [prediction_batch['ML_Prediction']==0,prediction_batch['ML_Prediction']==1]
        choices = ['Low','Medium']
        prediction_batch['Risk'] = np.select(conditions,choices,"High")
        prediction_batch['patient_id'] = range(len(df))
        prediction_batch = prediction_batch[['patient_id','Risk']]
        print(prediction_batch)
    else:
        return html.H3("Please provide inputs above")

    output = html.Div([html.H3("Probability for high risk mortality"),
                       html.Br(),
                       dt.DataTable(prediction_batch.to_dict('records'),
                                    [{"name": i, "id": i} for i in prediction_batch.columns],
                                    id='tbl', fixed_columns={'headers': True, 'data': 1},
                                    style_table={'minWidth': '100%'}, style_cell={'textAlign': 'center'},
                                    style_header={
                                        'backgroundColor': 'white',
                                        'fontWeight': 'bold'
                                    }
                                    ),
                       ])
    return output

@dash_app.callback(
    Output('prediction output2', 'children'),
    Input('submit-val', 'n_clicks'),
    State('Age', 'value'),
    State('SystolicBP', 'value'),
    State('DiastolicBP', 'value'),
    State('BS', 'value')
)
def update_output(n_clicks, Age, SystolicBP, DiastolicBP, BS):
    df_x = pd.DataFrame({'Age':[Age],'SystolicBP':[SystolicBP],
                         'DiastolicBP':[DiastolicBP],'BS':[BS]})
    prediction = score_model(df_x)
    print(prediction)
    prediction = prediction['predictions'][0]
    print(prediction)
    # compute SHAP values

    if prediction == 0:
        output = html.Div([html.H3("Mortality risk is low for patient"),
                           html.Br(),
                           ])
    elif prediction == 1:
        output = html.Div([html.H3("Mortality risk is moderate for patient"),
                           html.Br(),
                           ])
    else:
        output = html.Div([html.H3("Mortality risk is High for patient"),
                           html.Br(),
                           ])
    return output
### Run the App ###############################################
if __name__ == '__main__':
    dash_app.run_server(debug=True)