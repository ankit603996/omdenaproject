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
### Setup ###################################################
dash_app = dash.Dash(__name__,prevent_initial_callbacks=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
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
    prediction = clf.predict(df_x)[0]
    # compute SHAP values

    exp = explainer.explain_instance(df_x.values[0], clf.predict_proba , num_features=4)
    exp_df = pd.DataFrame(exp.as_list())
    exp_df.columns = ["Feature falling under rule","Impact_Amount"]
    exp_df['Impact'] = np.where(exp_df.Impact_Amount>0,"Positive_Imapct","Negative_Impact")
    if prediction == 0:
        output = html.Div([html.H3("Mortality risk is low for patient"),
                           html.Br(),
                           html.H5("Explaination for prediction"),
                           dt.DataTable(exp_df.to_dict('records'),
                                        [{"name": i, "id": i} for i in exp_df.columns],
                                        id='tbl', fixed_columns={'headers': True, 'data': 1},
                                        style_table={'minWidth': '100%'}, style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {
                                                'if': {'column_id': 'Region'},
                                                'textAlign': 'left'
                                            }
                                        ])
                           ])
    elif prediction == 1:
        output = html.Div([html.H3("Mortality risk is moderate for patient"),
                           html.Br(),
                           html.H5("Explaination for prediction"),
                           dt.DataTable(exp_df.to_dict('records'),
                                        [{"name": i, "id": i} for i in exp_df.columns],
                                        id='tbl', fixed_columns={'headers': True, 'data': 1},
                                        style_table={'minWidth': '100%'}, style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {
                                                'if': {'column_id': 'Region'},
                                                'textAlign': 'left'
                                            }
                                        ])
                           ])
    else:
        output = html.Div([html.H3("Mortality risk is High for patient"),
                           html.Br(),
                           html.H5("Explaination for prediction"),
                           dt.DataTable(exp_df.to_dict('records'),
                                        [{"name": i, "id": i} for i in exp_df.columns],
                                        id='tbl', fixed_columns={'headers': True, 'data': 1},
                                        style_table={'minWidth': '100%'},    style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {
                                                'if': {'column_id': 'Region'},
                                                'textAlign': 'left'
                                            }
                                            ])
                           ])
    return output
### Run the App ###############################################
if __name__ == '__main__':
    dash_app.run_server(debug=True)