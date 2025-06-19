# Chatbot initialization
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

# Use local Mistral model via Ollama
llm = ChatOllama(model="mistral", temperature=0)

# Vector DB with FAISS and indexing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup modello di embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2') # It maps sentences to a 384-dimensional vector space.

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=300):
    """
    Split the text into smaller chunks for embedding.

    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    The second chunk is has 300 characters of overlap with the first chunk.
    The third chunk has 300 characters of overlap with the second chunk, and so on.
    NOTICE: it is not garanteed that the first part of the second chunk is in the first chunk. The reason is that we split according to the separators breaks and not according to the characters.


    RecursiveCharacterTextSplitter splits the text by trying different separators recursively (in this case, first by \n\n, then by \n, then by ., then by space, and finally by empty string).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

with open("medquad_QA.txt", "r",encoding="utf-8") as file:
    document_text = file.read()

# Splitting in chunks and embedding
chunks = split_text(document_text)
chunk_embeddings = embedder.encode(chunks)

# Building of FAISS embedding db
dimension = chunk_embeddings.shape[1]  # embedding dimension
faiss_db = faiss.IndexFlatL2(dimension)
faiss_db.add(np.array(chunk_embeddings))

# Search function
def search(query, query_translation=True, k=2):
    """
    Search for the most relevant chunks in the FAISS database based on the query.
    Args:
        query (str): The query string to search for.
        query_translation (bool): if true, the chatbot gets the user question and other three questions which are similar to the user one but with different perspectives.
        k (int): The number of nearest neighbors to return.
    """
    if query_translation:
        template_multiple_questions = """You are an AI language model assistant. Your task is to generate two different versions of the given user question to retrieve relevant documents from a vector database. By generating
        multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original
        question: {question}"""
        prompt_multiple_questions = ChatPromptTemplate.from_template(template_multiple_questions)
        chain = prompt_multiple_questions | llm
        response_multiple_questions = chain.invoke({"question": query})
        questions = response_multiple_questions.content.split("\n")
        queries_embeddings = [embedder.encode([q]) for q in questions if q.strip() != ""]
        queries_embeddings.append(embedder.encode([query]))  # Append the original query embedding
        distances, indices = [], []
        for query_embedding in queries_embeddings:
            distances_, indices_ = faiss_db.search(np.array(query_embedding), k)
            distances.append(distances_)
            indices.append(indices_)
        indices = np.vstack(indices).reshape(1, k*3)
        results = [chunks[i] for i in list(set(indices[0]))]
    else:
        query_embedding = embedder.encode([query])
        distances, indices = faiss_db.search(np.array(query_embedding), k)
        results = [chunks[i] for i in list(set(indices[0]))]
    return results

# Chunk retrieving from the FAISS database
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    template = """Answer based ONLY on the provided context.

    === CONTEXT START ===
    {context}
    === CONTEXT END ===

    Instructions:

    1-3 sentences maximum.
    - If unsure, say exactly: 'Unknown according to current medical consensus.'

    Question: {question}
    Answer:"""
    question = request.form['question']  # Get the user input from the POST request
    results = search(query=question, query_translation=True, k=2)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"context": results, "question": question})
    answer_final = response.content 
    return jsonify({'answer': answer_final})  # Return the predicted answer as JSON


if __name__ == '__main__':
    app.run(debug=True)
