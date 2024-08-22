from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3

import pickle
import os
import logging
from datetime import datetime

# Set up logging
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"embedding_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

def save_embedding_database(embedding_db, file_name):
    """
    Save the embedding database to disk with error handling and logging.
    """
    try:
        logging.info(f"Starting to save embedding database to {file_name}")
        
        # Check if embedding_db is empty, handling both dict and numpy array cases
        if isinstance(embedding_db, dict):
            if not embedding_db:
                raise ValueError("Embedding database dictionary is empty")
        elif isinstance(embedding_db, np.ndarray):
            if embedding_db.size == 0:
                raise ValueError("Embedding database numpy array is empty")
        else:
            raise TypeError("Embedding database must be a dictionary or numpy array")

        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, file_name)
        
        with open(file_path, 'wb') as f:
            pickle.dump(embedding_db, f)
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise IOError(f"The saved file {file_path} is empty")
        
        logging.info(f"Embedding database successfully saved to {file_path}")
        logging.info(f"File size: {file_size} bytes")
        return True
    
    except ValueError as ve:
        logging.error(f"ValueError occurred: {ve}")
    except IOError as ioe:
        logging.error(f"IOError occurred: {ioe}")
    except TypeError as te:
        logging.error(f"TypeError occurred: {te}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while saving embedding database: {e}")
    
    return False

with open("trees.txt", "r", encoding="utf-8") as file:
    sentences = file.read()

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
embeddings = model.encode(sentences)
np.save('embeddings.npy',embeddings)

logging.info("Starting embedding process")

loaded_embeddings = np.load('embeddings.npy')
print("Loaded Embeddings:",loaded_embeddings)

        # Create a sample embedding database
logging.info(f"Created sample embedding database with {len(loaded_embeddings)} entries")

        # Save the database
save_success = save_embedding_database(loaded_embeddings, "C:/Users/Home/Desktop/embedding.pk1")
if save_success:
    logging.info("Database saved successfully")
else:
    logging.warning("Failed to save database")

