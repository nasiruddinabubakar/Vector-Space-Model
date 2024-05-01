# Import necessary libraries
import os
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Ensure the NLTK resources are available
import nltk


# Define a function to preprocess the query and documents
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define a function to preprocess the query and documents
def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Tokenize the text using NLTK
    tokens = word_tokenize(text)
    
    # Define a regular expression pattern for alphabetic words
    pattern = re.compile(r'^[a-z]+$')
    
    # Remove stopwords and keep only tokens that match the regex pattern
    filtered_tokens = [
        token for token in tokens
        if (token not in stopwords.words('english') and 
            3 < len(token) < 13 and 
            pattern.match(token))
    ]
    
    return filtered_tokens

# Define a function to calculate the term frequency (TF) for a text
def calculate_term_frequency(tokens):
    """Calculate term frequency (TF) for a given list of tokens."""
    # Initialize a dictionary to hold term frequencies
    term_frequency = defaultdict(int)
    
    # Calculate term frequencies
    for token in tokens:
        term_frequency[token] += 1
    
    # Convert the dictionary to a standard dictionary and return
    return dict(term_frequency)

# Define a function to calculate inverse document frequency (IDF)
def calculate_idf(documents):
    """Calculate inverse document frequency (IDF) for a list of documents."""
    # Initialize a dictionary to hold document frequency
    doc_frequency = defaultdict(int)
    
    # Calculate document frequency for each term
    for tokens in documents:
        # Use a set to get unique terms in the document
        unique_terms = set(tokens)
        for term in unique_terms:
            doc_frequency[term] += 1
    
    # Total number of documents
    N = len(documents)
    print(unique_terms.__len__());
    # Calculate IDF for each term
    idf = {}
    count = 0;
    for term, df in doc_frequency.items():
        # Calculate IDF using the formula
        count = count+1;
        idf[term] = math.log(N / df)
    print(count);
    return idf

# Define a function to calculate TF-IDF for a document
def calculate_tf_idf(tf, idf):
    """Calculate TF-IDF for a document."""
    tf_idf = {}
    for term, tf_value in tf.items():
        # Calculate TF-IDF using the formula
        tf_idf[term] = tf_value * idf.get(term, 0)
    return tf_idf

# Define a function to calculate cosine similarity between vectors
def calculate_cosine_similarity(query_vector, doc_vector):
    """Calculate cosine similarity between query and document vectors."""
    # Calculate the dot product
    dot_product = 0
    for term in query_vector:
        # Multiply query and document vector values and sum them
        dot_product += query_vector.get(term, 0) * doc_vector.get(term, 0)
    
    # Calculate the magnitude (Euclidean norm) of query and document vectors
    query_magnitude = np.sqrt(sum(value ** 2 for value in query_vector.values()))
    doc_magnitude = np.sqrt(sum(value ** 2 for value in doc_vector.values()))
    
    # Calculate cosine similarity using the formula
    if query_magnitude == 0 or doc_magnitude == 0:
        # If either vector has zero magnitude, similarity is zero (no correlation)
        return 0.0
    
    cosine_similarity = dot_product / (query_magnitude * doc_magnitude)
    
    return cosine_similarity

# Define a function to find the most relevant documents for a query
def find_relevant_documents(query, doc_vectors, idf, doc_filepaths):
    """Find the most relevant documents for a given query"""
    # Preprocess the query
    query_tokens = preprocess_text(query)
    
    # Calculate query TF and TF-IDF
    query_tf = calculate_term_frequency(query_tokens)
    query_tfidf = calculate_tf_idf(query_tf, idf)
    
    # Calculate similarity scores for each document
    similarity_scores = []
    for i, (doc_vector, filepath) in enumerate(zip(doc_vectors, doc_filepaths)):
        similarity = calculate_cosine_similarity(query_tfidf, doc_vector)
        similarity_scores.append((filepath, similarity))
    
    # Sort documents by similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return the sorted list of documents and their similarity scores
    return similarity_scores

   
        
        
@app.route('/search', methods=['POST'])
def get_vector():
    # Get the input query from the JSON request body
    input_query = request.json.get('query', '')
    if not input_query:
        # Return an error response if the query is invalid
        return jsonify({'error': 'Invalid input query'}), 400
    
    # Calculate relevant documents based on the input query
    relevant_docs = find_relevant_documents(input_query, tf_idf_vectors, idf, doc_filepaths)
    
    # Prepare the response data
    response_data = []
    for filepath, similarity in relevant_docs:
        # Filter out low similarity scores if necessary
        if similarity < 0.03:
            continue
        # Append each document and similarity score as a dictionary
        response_data.append({
            'document': filepath,
            'similarity': similarity
        })
    
    # Return the list of relevant documents as JSON
    return jsonify(response_data)

      # Define the paths to the text files (assume there are 10 document files in an array)
base_path = os.path.abspath(os.path.dirname(__file__))
file_paths = [os.path.join(base_path, 'app', 'data', f'{i+1}.txt') for i in range(26)]

    # Lists to store document tokens, term frequencies, and file paths
doc_tokens_list = []
term_frequencies = []
doc_filepaths = []
    
    # Read and preprocess the content of each document
for file_path in file_paths:
    try:
        with open(file_path, 'r') as file:
            # Preprocess the text and store the tokens
            doc_tokens = preprocess_text(file.read())
            doc_tokens_list.append(doc_tokens)
            
            # Calculate term frequency for each document and store it
            tf = calculate_term_frequency(doc_tokens)
            term_frequencies.append(tf)
            
            # Store the file path
            doc_filepaths.append(file_path)
    
    except FileNotFoundError:
        # Handle missing files by printing a warning message
        print(f"Warning: File not found - {file_path}")
        continue
    
    except Exception as e:
        # Handle any other unexpected errors and print an error message
        print(f"Error processing file {file_path}: {e}")
        continue

# Calculate inverse document frequency (IDF) for the terms across all documents
idf = calculate_idf(doc_tokens_list)

# Calculate TF-IDF for each document and store them in a list
tf_idf_vectors = []
for tf in term_frequencies:
    tf_idf = calculate_tf_idf(tf, idf)
    tf_idf_vectors.append(tf_idf)
    
# Run the main function
if __name__ == "__main__":
    app.run(debug=True)


