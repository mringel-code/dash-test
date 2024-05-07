from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import datetime
import io
import os

import pandas as pd

#new imports
from sagemaker.predictor import retrieve_default
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-3-8b-instruct"
from typing import Dict, List
import boto3

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    dialog = [
        {"role": "system", "content": "You are a business assistant for the insurance industry, skilled in summarizing complex requests for proposal (RfP) with highest precision."},
        {"role": "user", "content": f"Summarize the following text:\n This is a dummy text. It doesnt say anything."},
        {"role": "assistant", "content": ""}
    ]
    prompt = format_messages(dialog)
    payload_in = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 640,
            "top_p": 0.9,
            "temperature": 0.6,
            "stop": "<|eot_id|>"
        }
    }
    
    client = boto3.client('sagemaker-runtime')
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
    endpoint_name = "jumpstart-dft-meta-textgeneration-llama-3-8b-instruct"# Your endpoint name.
    content_type = "application/json"  # The MIME type of the input data in the request body.
    accept = "Accept" # The desired MIME type of the inference in the response.
    payload = payload_in # Payload for inference.
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        CustomAttributes=custom_attributes, 
        ContentType=content_type,
        Accept=accept,
        Body=payload
    )

    response = response['Body'].read().decode('utf-8')
    
    try:
        #with open(filename, "wb") as fp:   # Unpacks the uploaded files
            #fp.write(decoded)   
        if 'pdf' in filename:  # Check if it is a pdf file
            file_path = os.path.abspath(filename)
            #result = application.main(file_path)  # Process with another script
            #summary = generate_summary("This is a dummy text. It doesnt say anything.")
            result = summary
            
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])
    
    return html.Div([
                html.H5(filename),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.P(str(result))
            ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#Added code
def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for Llama-3 chat models.
    """
    prompt: List[str] = []
    prompt.extend(["<|begin_of_text|>"])

    if messages[0]["role"] == "system":
        content = "".join(["<|start_header_id|>system<|end_header_id|>\n\n", messages[0]["content"], "<|eot_id|>", "<|start_header_id|>user<|end_header_id|>\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, assistant in zip(messages[::2], messages[1::2]):
        prompt.extend(["<|start_header_id|>user<|end_header_id|>\n\n", (user["content"]).strip(), "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", (assistant["content"]).strip()])

    return "".join(prompt)

def generate_summary(text):
    dialog = [
        {"role": "system", "content": "You are a business assistant for the insurance industry, skilled in summarizing complex requests for proposal (RfP) with highest precision."},
        {"role": "user", "content": f"Summarize the following text:\n" + text},
        {"role": "assistant", "content": ""}
    ]
    prompt = format_messages(dialog)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 640,
            "top_p": 0.9,
            "temperature": 0.6,
            "stop": "<|eot_id|>"
        }
    }
    predictor = retrieve_default(endpoint_name)
    response = predictor.predict(payload)
    return response["generated_text"]

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8080)
