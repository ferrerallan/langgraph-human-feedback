# QA Feedback System

A human-in-the-loop learning system for question answering with continuous feedback and improvement.

## Overview

This system provides an interactive question-answering workflow that:

1. Generates responses to user questions using LLMs
2. Searches for similar previously validated questions in a vector database
3. Adapts existing responses when similar questions are found
4. Collects human feedback on response quality
5. Regenerates responses based on feedback when needed
6. Stores validated responses for future use

The system is built using LangGraph, which provides a framework for creating complex workflows with LLMs and human feedback.

## Architecture

The application uses a directed graph architecture to manage the workflow:

```
START → Generate Response → Human Feedback → Evaluate Feedback → [Save Response or Regenerate]
```

### Key Components

- **State Management**: Typed state definition that tracks all relevant information throughout the process
- **Vector Database**: Stores and retrieves validated responses using semantic similarity
- **Language Model Service**: Handles interactions with the LLM to generate and adapt responses
- **Human Feedback Loop**: Interrupts the flow to collect validation and improvement suggestions
- **Conditional Routing**: Determines next steps based on feedback evaluation

## Features

- ✅ Semantic search for similar questions
- ✅ Automated response adaptation for similar questions
- ✅ Human validation of all responses before storage
- ✅ Multiple regeneration attempts (up to 3) based on specific feedback
- ✅ Persistence of validated responses in vector database
- ✅ Visualization of the workflow graph
- ✅ Checkpoint system to resume interrupted workflows

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/qa-feedback-system.git
   cd qa-feedback-system
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

Run the main script to start the interactive QA system:

```bash
# If using Poetry
poetry run python main.py

# Or activate the virtual environment first
poetry shell
python main.py
```

The system will prompt you to enter questions. For each question:

1. The system will search for similar questions or generate a new response
2. You will be asked to validate the response (yes/no)
3. If rejected, you can provide feedback for improvement
4. The system will regenerate up to 3 times based on your feedback
5. Once validated, the response is stored for future use

## Configuration

The system is configured in `config.py`:

```python
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
```

Key configuration options:

- `LLM_MODEL`: Language model to use (default: "gpt-4")
- `VECTOR_DB_PATH`: Location for the vector database (default: "./chroma_db")
- `SQLITE_DB_PATH`: Location for the checkpoint database (default: "checkpoints.sqlite")
- `SIMILARITY_THRESHOLD`: Threshold for considering questions similar (default: 0.5)
- `MAX_SIMILAR_RESULTS`: Maximum number of similar results to retrieve (default: 2)
- `MAX_RETRY_ATTEMPTS`: Maximum number of regeneration attempts (default: 3)

## Project Structure

```
qa-feedback-system/
├── config.py                  # Configuration settings
├── main.py                    # Entry point
├── graph/
│   ├── builder.py             # Graph construction
│   └── state.py               # State definition
├── nodes/
│   ├── evaluate.py            # Feedback evaluation node
│   ├── generate_response.py   # Response generation node
│   ├── human_feedback.py      # Human feedback collection node
│   ├── regenerate.py          # Response regeneration node
│   └── storage.py             # Response storage node
├── services/
│   ├── llm_service.py         # LLM interaction service
│   ├── vector_db.py           # Vector database service
│   └── visualization.py       # Graph visualization service
└── utils/
    └── helpers.py             # Helper functions
```

## Development

To extend the system:

1. Add new nodes in the `nodes/` directory
2. Update the graph structure in `graph/builder.py`
3. Extend the state definition in `graph/state.py` if needed
4. Add or modify services in the `services/` directory

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
