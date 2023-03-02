# app.py

from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import chromadb
import pickle
import openai

import os
os.environ['OPENAI_API_KEY'] = "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"

# Load the QA chain model from the pickle file
with open('qa_chain.pkl', 'rb') as f:
    chain = pickle.load(f)

with open('text.txt') as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

class MyDocument:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata
        self.page_content = text

metadata = [{'source': str(i)} for i in range(len(texts))]
documents = [MyDocument(text, metadata=metadata[i]) for i, text in enumerate(texts)]
docsearch = Chroma.from_documents(documents, embeddings)

# Define the Flask app
app = Flask(__name__, template_folder = 'templates')

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = ''
    if request.method == 'POST':
        query = request.form['query']
        docs = docsearch.similarity_search(query, k=1)
        response_text = response(query, docs, chain)
    return render_template('index.html', answer=response_text)

# Define the response function to generate a response using the QA model
def response(query, docs, chain):
    result = chain.run(input_documents=docs, question=query)
    if result.strip() == "I don't know.":
        response = openai.Completion.create(
            model='davinci',
            prompt=query,
            temperature=0.0,
            max_tokens=20,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
        )
        response_text = response.choices[0].text.strip()
    else:
        response_text = result.strip()
    return response_text

if __name__ == '__main__':
    app.run(debug=True)

