import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Read JSON file
with open('intents.json', 'r') as file:
    data = json.load(file)

# Initialize lists to store preprocessed data
patterns_processed = []
responses = []
tags = []

# Tokenize patterns, lowercase, remove punctuation, and stopwords
for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lowercase
        tokens = word_tokenize(pattern.lower())
        # Remove punctuation
        tokens = [word for word in tokens if word.isalnum()]
        # Optionally remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        patterns_processed.append(tokens)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Save preprocessed data
preprocessed_data = {
    'patterns': patterns_processed,
    'responses': responses,
    'tags': tags
}

with open('preprocessed_data_tag.json', 'w') as file:
    json.dump(preprocessed_data, file)
