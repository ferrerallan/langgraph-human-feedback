import logging
import os
import sys
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create logger
logger = logging.getLogger("qa_feedback_system")
logger.setLevel(logging.ERROR)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)

# Create file handler
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(f"{log_dir}/qa_feedback_{timestamp}.log")
file_handler.setLevel(logging.DEBUG)

# Create formatters
console_format = logging.Formatter('%(levelname)s: %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set formatters
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def log_question(question: str):
    """Log a new question."""
    logger.info(f"New question: {question}")

def log_response(response: str, source: str):
    """Log a response with its source."""
    logger.info(f"Response from {source}: {response[:100]}...")
    logger.debug(f"Full response from {source}: {response}")

def log_feedback(valid: bool, feedback: str = ""):
    """Log user feedback."""
    if valid:
        logger.info("Response validated")
    else:
        logger.info(f"Response rejected. Feedback: {feedback}")

def log_error(error: Exception, context: str = ""):
    """Log an error."""
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")