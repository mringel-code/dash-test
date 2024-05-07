from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import datetime
import io
import os

import pandas as pd

#new imports
from typing import Dict, List
import boto3
import json
from PyPDF2 import PdfReader

client = boto3.session.Session().client('sagemaker-runtime')
endpoint_name = 'jumpstart-dft-meta-textgeneration-llama-3-8b-instruct' # Your endpoint name.
content_type = 'application/json'  # The MIME type of the input data in the request body.

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
    #try:
        #if 'pdf' in filename:  # Check if it is a pdf file
    file_path = os.path.abspath(filename)
    result = summarize_data(file_path)
            
    #except Exception as e:
        #print(e)
        #return html.Div(['There was an error processing this file.'])
    
    return html.Div([
                html.H5(file_path),
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
# Function to summarize the text of a pdf in a given file path
def summarize_data(file_path):
    # Path to the large pdf
    large_text = extract_text_from_pdf(file_path)
    # Split text into chunks
    chunk_size = 3072
    text_chunks = [large_text[i:i + chunk_size] for i in range(0, len(large_text), chunk_size)]
    # Generate a summary for each chunk
    summaries = [generate_summary(chunk) for chunk in text_chunks]
    # Generate the final summary from the summaries of chunks
    final_summary = generate_summary(" ".join(summaries))
    #log_results(final_summary)
    return final_summary

# Function to extract text from pdf
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    count = len(reader.pages)
    for i in range(count):
        page = reader.pages[i]
        text += page.extract_text()
    return text
    
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
    payload_in = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 640,
            "top_p": 0.9,
            "temperature": 0.6,
            "stop": "<|eot_id|>"
        }
    }
    
    payload = json.dumps(payload_in) # Payload for inference.
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType=content_type,
        Body=payload
    )
    return response['Body'].read().decode('utf-8')

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8080)
