import openai

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(override=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

LLM_MODEL = "gpt-4"

VECTOR_DB_PATH = "./chroma_db"

SQLITE_DB_PATH = "checkpoints.sqlite"

SIMILARITY_THRESHOLD = 0.5
MAX_SIMILAR_RESULTS = 2
MAX_RETRY_ATTEMPTS = 3