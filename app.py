from dash import Dash, dcc, html, dash_table, Input, Output, State, callback

import base64
import datetime
import io
import os
import shutil

import pandas as pd

#new imports
from typing import Dict, List
import boto3
import json
from PyPDF2 import PdfReader
from io import BytesIO

#RAG imports
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

client = boto3.session.Session().client('sagemaker-runtime')

endpoint_name = 'jumpstart-dft-meta-textgeneration-llama-3-8b-instruct' # Your endpoint name.
content_type = 'application/json'  # The MIME type of the input data in the request body.

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

CHROMA_PATH = "chroma"

import langchain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

aws_region = boto3.session.Session().region_name

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

emb_content_handler = ContentHandler()

embeddings = SagemakerEndpointEmbeddings(
    endpoint_name='jumpstart-dft-hf-textembedding-gpt-j-6b-fp16',
    region_name= aws_region,
    content_handler=emb_content_handler,
)

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
    text = ""
    reader = PdfReader(BytesIO(decoded))
    count = len(reader.pages)
    for i in range(count):
        page = reader.pages[i]
        text += page.extract_text()
    summary_result = summarize_data(text)
    
    text = "Hi! It's time for the beach"
    text_embedding = embeddings.embed_query(text)
    summary_result = summary_result + str(text_embedding[:5])
    
    #loader = PyPDFLoader(BytesIO(decoded))
    #chunks = loader.load_and_split()
    chunks = reader.pages
    save_to_chroma(chunks)
    #query_result = query_data("What is the deadline for the RfP?")
                
    #except Exception as e:
        #print(e)
        #return html.Div(['There was an error processing this file.'])
    
    return html.Div([
                html.H5(file_path),
                html.H6(datetime.datetime.fromtimestamp(date)),
                html.P(str(summary_result))
                #html.P(str(query_result))
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
def summarize_data(text):
    # Split text into chunks
    chunk_size = 3072
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    # Generate a summary for each chunk
    summaries = [generate_summary(chunk) for chunk in text_chunks]
    # Generate the final summary from the summaries of chunks
    final_summary = generate_summary(" ".join(summaries))
    #log_results(final_summary)
    return final_summary
    
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
        {"role": "system", "content": f"You are a business assistant for the insurance industry, skilled in summarizing a given request for proposal (RfP) with highest precision."},
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


def save_to_chroma(chunks: List[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    collection_metadata = {"hnsw:space": "cosine"} # Define the metadata to change the distance function to cosine
    
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding, persist_directory=CHROMA_PATH, collection_metadata=collection_metadata
    )
    db.persist()
    #print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

"""
def query_data(query_text):
    # Prepare the DB.
    embedding_function = lc_embed_model
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.1: # original: < 0.7
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    response_text = generate_response(context=context_text, question=query_text)
    #sources = [doc.metadata.get("source", None) for doc, _score in results]
    #formatted_response = f"Response: {response_text}"
    return response_text
  
def generate_response(context, text):
    dialog = [
        {"role": "system", "content": "You are a business assistant for the insurance industry, skilled in answering questions for a given request for proposal (RfP) with highest precision. Answer the question based only on the following context:\n\n" + context},
        {"role": "user", "content": f"Answer the question based on the above context:\n" + text},
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
"""

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8080)
