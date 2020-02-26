import basilica

import os
from dotenv import load_dotenv

load_dotenv()

def basilica_api():
    API_KEY = os.getenv("BASILICA_API_KEY")
    connection = basilica.Connection(API_KEY)
    return connection
# embeddings = connection.embed_sentences(["Hello world!", "How are you?"])
# print(list(embeddings)) # [[0.8556405305862427, ...], ...]