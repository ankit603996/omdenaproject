#### Import Packages ########################################
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
### Setup ###################################################
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app.title = 'Machine Learning Model Deployment'
app = dash_app.server

### load ML model ###########################################
with open('xgb_model.pkl', 'rb') as f:
    clf = pickle.load(f)
### load explainer ###########################################
with open('explainer', 'rb') as f:
    explainer = dill.load(f)
### App Layout ###############################################
dash_app.layout = html.Div([
    html.H1("Omdena: MobileUurka Risk Prediction Model", style={'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.Div([
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
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')))
    return df

### Callback to produce the prediction #########################
@dash_app.callback(
    Output('prediction output', 'children'),
    Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
    Input('submit-val', 'n_clicks'),
)

def update_output(list_of_contents, list_of_names, list_of_dates,n_click):
    if list_of_contents is not None:
        df_list = [parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        df = df_list[0]
        print("df type",type(df))
        print("df shape", df.shape)
        data = df.copy()
        prediction_batch = pd.DataFrame(clf.predict_proba(data))
        prediction_batch.columns = ['prob for high risk', 'prob for medium risk', 'prob for low risk']
        prediction_batch['patient_id'] = range(len(df))
        prediction_batch = prediction_batch[['patient_id','prob for high risk', 'prob for medium risk', 'prob for low risk']]
        print(prediction_batch)
    else:
        return html.H3("Please provide inputs above")

    fig = px.histogram(prediction_batch['prob for high risk'], x="prob for high risk", histnorm='probability density')

    # compute SHAP values
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
                       dcc.Graph(figure=fig)
                       ])
    return output


### Run the App ###############################################
if __name__ == '__main__':
    dash_app.run_server(debug=True)