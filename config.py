import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

# Vector database configuration
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")

# Feedback system configuration
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Checkpoint configuration
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")