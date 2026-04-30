# Python Agent

This project demonstrates various patterns for building AI agents with Python and OpenAI. The examples showcase how to structure agentic workflows, from basic API calls to more complex patterns like prompt chaining, routing, and parallelization.

## Project Structure

The project is organized into two main directories:

- `basics/`: This directory contains fundamental examples of using the OpenAI API.
- `workflow_patterns/`: This directory showcases more advanced agentic workflows.

### `basics/`

- `basic.py`: A simple example of how to make a chat completion call to the OpenAI API.
- `knowledge_base.json`: A sample knowledge base used by the `retrieval.py` example.
- `retrieval.py`: Demonstrates how to use a knowledge base to answer questions.
- `structured.py`: Shows how to get structured output from the model using Pydantic.
- `tools.py`: An example of how to use tools with the OpenAI API.

### `workflow_patterns/`

- `parallelization.py`: An example of how to run multiple API calls in parallel to validate a user's request.
- `prompt_chaining.py`: Demonstrates how to chain multiple prompts together to perform a complex task.
- `routing.py`: Shows how to route a user's request to the appropriate tool or function.

## Setup and Testing

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone --filter=blob:none --sparse https://github.com/kaitosuzuki-CS/practice.git
    cd practice
    git sparse-checkout set agent_basics
    cd agent_basics
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root of the project and add your OpenAI API key:

    ```
    OPENAI_API_KEY=your-api-key
    ```

### Running the Examples

You can run any of the examples by executing the Python scripts directly. For example, to run the basic example:

```bash
python basics/basic.py
```

To run the prompt chaining example:

```bash
python workflow_patterns/prompt_chaining.py
```

### Testing

Each script is self-contained and can be run individually to test its functionality. There are no separate test files for the examples.
