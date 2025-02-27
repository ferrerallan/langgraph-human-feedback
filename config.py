import os
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

LLM_MODEL = "gpt-4"

VECTOR_DB_PATH = "./chroma_db"

SQLITE_DB_PATH = "checkpoints.sqlite"

SIMILARITY_THRESHOLD = 0.5
MAX_SIMILAR_RESULTS = 2
MAX_RETRY_ATTEMPTS = 3