import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


document_1 = {
    "id": 1,
    "title": "Introduction to Drones and UAV Technology",
    "category": "Drones",
    "text": """A drone is an unmanned aerial vehicle (UAV), which means it is an aircraft that operates without a human pilot onboard. 
    Drones can be controlled remotely by an operator or fly autonomously using onboard sensors, GPS, and AI-based algorithms. 
    Drones are used for aerial photography and mapping. Agricultural drones help farmers spray pesticides efficiently. 
    Drone technology is advancing rapidly in military applications.""",
    "text_length": 359,
    "key_terms": ["drone", "unmanned aerial vehicle", "UAV", "AI-based algorithms", "aerial photography", "agriculture", "military applications"],
    "word_count": 68,
    "keywords_count": {
        "drone": 3,
        "UAV": 1,
        "AI": 1,
        "agriculture": 1,
        "military": 1
    },
    "use_cases": ["aerial photography", "mapping", "agriculture", "military"],
    "sentiment": "neutral"
}

document_2 = {
    "id": 2,
    "title": "Overview of Agriculture",
    "category": "Agriculture",
    "text": """Agriculture is the practice of cultivating soil, growing crops, and raising animals for food, fiber, medicinal plants, and other 
    products used to sustain and enhance human life. It has been the foundation of human civilization for thousands of years and plays 
    a vital role in the global economy, providing essential resources for food, energy, and raw materials.""",
    "text_length": 377,
    "key_terms": ["agriculture", "cultivating", "food", "fiber", "human civilization", "global economy"],
    "word_count": 68,
    "keywords_count": {
        "agriculture": 2,
        "food": 1,
        "fiber": 1,
        "economy": 1
    },
    "use_cases": ["farming", "food production", "fiber", "raw materials"],
    "sentiment": "positive"
}

document_3 = {
    "id": 3,
    "title": "What is Sport?",
    "category": "Sports",
    "text": """Sport refers to physical activities or games, often competitive in nature, that involve skill, strategy, and physical exertion. 
    Sports can be played individually or as a team, and they typically have rules and guidelines governing play. They provide entertainment, 
    promote fitness, and are often used as a means of social interaction, helping people to build teamwork, leadership, and communication skills.""",
    "text_length": 386,
    "key_terms": ["sport", "competitive", "fitness", "teamwork", "leadership", "entertainment"],
    "word_count": 72,
    "keywords_count": {
        "sport": 3,
        "teamwork": 1,
        "fitness": 1,
        "leadership": 1
    },
    "use_cases": ["fitness", "entertainment", "teamwork", "leadership"],
    "sentiment": "positive"
}

document_4 = {
    "id": 4,
    "title": "Understanding Artificial Intelligence",
    "category": "AI",
    "text": """Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think, 
    learn, and perform tasks that typically require human cognitive processes. AI systems can be designed to solve problems, recognize 
    patterns, understand natural language, and make decisions, among many other capabilities. Over the years, AI has rapidly advanced, 
    transforming many industries, from healthcare to robotics, and playing a crucial role in daily life.""",
    "text_length": 464,
    "key_terms": ["artificial intelligence", "AI", "machine learning", "cognitive processes", "pattern recognition", "robotics"],
    "word_count": 84,
    "keywords_count": {
        "AI": 3,
        "intelligence": 1,
        "robotics": 1,
        "machine learning": 1
    },
    "use_cases": ["problem-solving", "healthcare", "robotics", "pattern recognition", "decision making"],
    "sentiment": "neutral"
}


documents = [document_1["text"], document_2["text"], document_3["text"], document_4["text"]]
labels = ["drones", "agriculture", "sports", "ai"]

tf_idf = TfidfVectorizer()
X_vectors = tf_idf.fit_transform(documents)

X_train, X_test, y_train, y_test = train_test_split(X_vectors, labels, test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)


def classify_text(query):
    query_vector = tf_idf.transform([query])  
    predicted_label = classifier.predict(query_vector)[0]
    

    similarity = cosine_similarity(query_vector, X_train).flatten()
    ranked_indices = similarity.argsort()[-3:][::-1]
    
    result = []
    for i in ranked_indices:
        result.append({
            'document': documents[i],
            'category': labels[i],
            'similarity_score': similarity[i]
        })


    accuracy = 1 if labels[ranked_indices[0]] == predicted_label else similarity[ranked_indices[0]] * 100

    return predicted_label, accuracy, result
