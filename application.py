import argparse
import os
import shutil
import csv
from langdetect import detect
from PyPDF2 import PdfReader
from sagemaker.predictor import retrieve_default
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

lc_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

endpoint_name = "jumpstart-dft-meta-textgeneration-llama-3-8b-instruct"
predictor = retrieve_default(endpoint_name)

CHROMA_PATH = "chroma"
DATA_PATH = "data/pdfs/"

def main(file_path):
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="The file.")
    args = parser.parse_args()
    file_path = DATA_PATH + args.file
    os.environ["FILE_PATH"] = file_path
    '''
    summarize_data(file_path)
    create_database()
    query_data("What is the deadline for the RfP?")
    query_data("What skills are required? Phrase your answer as a list of short skill labels.")
    # "What skills are required? Phrase your answer as a list of short skill labels."

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
    log_results(final_summary)
    print(final_summary)
    
# Function to extract text from pdf
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    count = len(reader.pages)
    for i in range(count):
        page = reader.pages[i]
        text += page.extract_text()
    return text

# Function to generate a summarization
# Phrase the summary in a way that it can be used to create a summary of multiple summary-parts in the enc.
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
    response = predictor.predict(payload)
    return response["generated_text"]

def log_results(final_summary):
    file_path = os.getcwd() + '/results.csv'
    current_datetime = datetime.now()

    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        is_empty = True
    else:
        is_empty = False

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if is_empty:
            writer.writerow(['Datetime','RfP Source', 'Summary', 'Model', 'Tokenizer', 'Code'])  # writes column name if the file is empty
        writer.writerow([current_datetime, os.environ.get("FILE_PATH"), final_summary, endpoint_name, "-", os.getcwd() + "/application.py"])  # writes the actual summary

def create_database():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = PyPDFLoader(os.environ.get("FILE_PATH"))
    documents = loader.load_and_split()
    return documents

def split_text(documents: list[Document]):
    chunks = documents
    print(f"Split document into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    collection_metadata = {"hnsw:space": "cosine"} # Define the metadata to change the distance function to cosine
    
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, lc_embed_model, persist_directory=CHROMA_PATH, collection_metadata=collection_metadata
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

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
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}"
    print(formatted_response)
    

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

# Function to generate a summarization
def generate_response(context, question):
    dialog = [
        {"role": "system", "content": f"Answer the question based only on the following context:\n\n" + context},
        {"role": "user", "content": f"\n---\nAnswer the question based on the above context: " + question},
        {"role": "assistant", "content": ""}
    ]
    prompt = format_messages(dialog)
    print(prompt)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 640,
            "top_p": 0.9,
            "temperature": 0.6,
            "stop": "<|eot_id|>"
        }
    }
    response = predictor.predict(payload)
    return response["generated_text"]

# Flask routes
@app.route('/', methods=['GET','POST'])
def upload_file():
    # Check if a POST request is made and it contains a file part.
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Your logic to call functions here...
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Your logic to get result here...
    return render_template('result.html', filename=filename)