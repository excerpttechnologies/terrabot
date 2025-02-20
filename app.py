# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# from keras.models import load_model
# import nltk
# from nltk.stem import WordNetLemmatizer
# import json
# import pickle
# import random

# app = Flask(__name__)
# CORS(app)

# # Initialize lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Load trained model and data
# model = load_model('chatbot_model.h5')
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))

# def clean_up_sentence(sentence):
#     # Tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # Lemmatize each word - create base word, in attempt to represent related words
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words):
#     # Tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # Bag of words - matrix of N words, vocabulary matrix
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 # Assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     # Filter below threshold predictions
#     ERROR_THRESHOLD = 0.25
    
#     # Generate probabilities from the model
#     bow_data = bow(sentence, words)
#     res = model.predict(np.array([bow_data]))[0]
    
#     # Filter out predictions below a threshold
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
#     # Sort by probability
#     results.sort(key=lambda x: x[1], reverse=True)
    
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
#     return return_list

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "I'm not sure how to respond to that. Could you please rephrase?"
    
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
    
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             # Get a random response from the intent
#             result = random.choice(i['responses'])
#             break
    
#     return result

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     try:
#         data = request.json
#         message = data.get('message', '')
        
#         # Get prediction
#         ints = predict_class(message)
        
#         # Get response
#         response = get_response(ints, intents)
        
#         # Return response and confidence
#         return jsonify({
#             'response': response,
#             'confidence': float(ints[0]['probability']) if ints else 0
#         })
    
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({
#             'response': "I encountered an error. Please try again.",
#             'confidence': 0
#         }), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import logging
import download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

try:
    # Load trained model and data
    logger.info("Loading model and data files...")
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents_v1.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    logger.info("Model and data files loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

@app.route('/api/chat', methods=['POST'])
def chat():
    logger.info("Received chat request")
    try:
        data = request.json
        logger.info(f"Request data: {data}")
        
        message = data.get('message', '')
        if not message:
            return jsonify({
                'response': "No message provided",
                'confidence': 0
            }), 400
        
        # Get prediction
        ints = predict_class(message)
        logger.info(f"Prediction complete. Intent: {ints}")
        
        # Get response
        response = get_response(ints, intents)
        logger.info(f"Generated response: {response}")
        
        return jsonify({
            'response': response,
            'confidence': float(ints[0]['probability']) if ints else 0
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I encountered an error. Please try again.",
            'confidence': 0
        }), 500

# Your existing helper functions remain the same
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    ERROR_THRESHOLD = 0.25
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that. Could you please rephrase?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
