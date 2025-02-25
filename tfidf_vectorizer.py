# Import necessary libraries
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents (corpus)
document_1 = "Tom loves coding in Python"
document_2 = "Jake loves coding in Java"
document_3 = "Emma enjoys programming in Python"

# Store documents in a list
documents = [document_1, document_2, document_3]

# Initialize TF-IDF Vectorizer
tf_idf = TfidfVectorizer()

# Fit and transform the documents into TF-IDF vectors
tf_idf_vector = tf_idf.fit_transform(documents)

# Function to compute similarity of a query with the documents
def get_query_vector(query, tf_idf, tf_idf_vector, documents, top_n=3):
    query_vector = tf_idf.transform([query])  # Convert query to TF-IDF vector
    similarity = cosine_similarity(query_vector, tf_idf_vector).flatten()  # Compute cosine similarity
    ranked_ind = similarity.argsort()[-top_n:][::-1]  # Get top matching documents
    return [(documents[i], similarity[i]) for i in ranked_ind]

# Query to compare with the existing documents
query = "I love Java but coding in Python"

# Get the most similar documents
top_results = get_query_vector(query, tf_idf, tf_idf_vector, documents)

# Print the results
print("\nTop matching documents:")
for doc, similarity in top_results:
    print(f"Document: {doc} | Similarity Score: {similarity:.4f}")
