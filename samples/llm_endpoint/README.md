# LLM Endpoint Sample

This sample project demonstrates how to create a custom LLM endpoint that can be used with SillyTavern or any other application that supports OpenAI-compatible APIs.

The endpoint provides an intelligent chat interface that:

- Automatically loads and indexes files from a `backstory/` directory, allowing the LLM to reference relevant background information during conversations
- Maintains conversation history beyond normal context limits by intelligently compressing past interactions and retrieving relevant context as needed
- Exposes an OpenAI-compatible API interface that works with any application supporting custom OpenAI-like model endpoints

The core functionality is implemented in:
- `__main__.py` - Sets up the FastAPI server and loads the backstory files
- `backstory_llm.py` - Implements the BackstoryLLMEngine that handles conversation memory and backstory integration

## Setup

1. Edit the following configuration files according to the [horsona README](https://github.com/synthbot-anon/horsona/blob/main/README.md):
   - `llm_config.json` - Configure LLM settings. This project requires a `reasoning_llm` engine for LLM calls.
   - `index_config.json` - Configure index settings. This project requires a `query_index` index for working with embeddings.
   - `.env` - Copy from `.env.example` and adjust it to set environment variables

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Add backstory files to the `backstory` directory.
   These should be text files containing information about the world that the character inhabits.
   On first run, these files will be summarized and indexed so they can be retrieved as necessary.

4. Run the application:
   ```bash
   poetry run python -m llm_endpoint
   ```

5. Run & configure SillyTavern to use the new endpoint. The custom LLM will be exposed on `http://localhost:8001/api/v1` as a model named `backstory-llm`.