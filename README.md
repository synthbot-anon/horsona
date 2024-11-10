# Horsona: The Swiss Army Knife of Pony Chatbot Creation
Creating a realistic pony chatbot is very difficult. This repo will try to maintain an organized collection of features that a pony chatbot might need in the hopes that future chatbot developers will have a easier time with it.

# Installation
Install poetry using the instructions from [python-poetry/install.python-poetry](https://github.com/python-poetry/install.python-poetry.org).

Install the repo:
```bash
# Clone the repo
git clone git@github.com:synthbot-anon/horsona.git

# Install dependencies and create the config files
cd horsona
./dev-install.sh
```

Configure the environment variables in `.env`:
```bash
# Edit .env to include all of your API keys.
# The default config requires an OpenAI API key.
# Example:
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxx"
```

# Running samples
Check out the [samples/](https://github.com/synthbot-anon/horsona/tree/main/samples) directory for example applications built with Horsona. Each sample has its own README with setup and usage instructions.


# Running tests
Most tests are stochastic since LLMs tend to be stochastic, so they may sometimes fail. Some tests may require additional configuration. If a test fails, check the README for that test to see if there are any additional steps you need to take. Most tests require the `llm_config.json` to include a `reasoning_llm` and the `index_config.json` to include a `query_index`.

You can run tests using `pytest`. For example:
```bash
# Run tests/test_llm.py.
poetry run pytest tests/llm/test_llm.py

# Run all tests.
poetry run pytest

# Run tests in parallel.
# You may run into API call limits doing this, so only use however many your API(s) will allow.
poetry run pytest -n 4
```

# Using a different LLM API
1. Edit `.env` to include your new LLM's API key(s).
2. Edit `llm_config.json` to use your new LLM(s). Supported "types" include:
   - AsyncCerebrasEngine
   - AsyncGroqEngine
   - AsyncFireworksEngine
   - AsyncOpenAIEngine
   - AsyncAnthropicEngine
   - AsyncTogetherEngine
   - AsyncPerplexityEngine

# Using multiple LLM APIs simultaneously (for speed)
1. Edit `.env` to include API keys for your additional LLMs.
2. Edit `llm_config.json` to include your LLMs each with a unique key.
3. Define a `reasoning_llm` in `llm_config.json` that uses the `MultiEngine` type with each LLM as an engine. Example:

```json
[
  {
    "cerebras_llama31_70b": {
      "type": "AsyncCerebrasEngine",
      "model": "llama3.1-70b",
      "rate_limits": [
        {"interval": 1, "max_calls": 3, "max_tokens": 240000},
        {"interval": 60, "max_calls": 120, "max_tokens": 240000},
        {"interval": 3600, "max_calls": 2600, "max_tokens": 4000000},
        {"interval": 86400, "max_calls": 57600, "max_tokens": 4000000}
      ]
    }
  },
  {
    "fireworks_llama31_70b": {
      "type": "AsyncFireworksEngine",
      "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
      "rate_limits": [
        {"interval": 1, "max_calls": 3, "max_tokens": null},
        {"interval": 60, "max_calls": 600, "max_tokens": null}
      ]
    }
  },
  {
    "openai_gpt4o_mini": {
      "type": "AsyncOpenAIEngine",
      "model": "gpt-4o-mini",
      "rate_limits": [
        {"interval": 60, "max_calls": 500, "max_tokens": 200000},
        {"interval": 86400, "max_calls": 10000, "max_tokens": null}
      ]
    }
  },
  {
    "anthropic_claude3_haiku": {
      "type": "AsyncAnthropicEngine",
      "model": "claude-3-haiku-20240307",
      "rate_limits": [
        {"interval": 60, "max_calls": 50, "max_tokens": 50000},
        {"interval": 86400, "max_calls": null, "max_tokens": 5000000}
      ]
    }
  },
  {
    "reasoning_llm": {
      "type": "MultiEngine",
      "engines": [
        "cerebras_llama31_70b",
        "fireworks_llama31_70b",
        "openai_gpt4o_mini",
        "anthropic_claude3_haiku"
      ]
    }
  }
]
```

# Using Ollama instead of OpenAI for embeddings
1. Run Ollama.

```bash
docker run -d --rm -e OLLAMA_HOST=0.0.0.0:11434 -v "ollama:/root/.ollama" -p 11434:11434 --name ollama ollama/ollama
```

> Note: You can stop Ollama with the following command: \
> Stop gracefully: `docker container kill ollama` \
> Stop forcefully: `docker container rm -f ollama`


2. Load an embedding model into Ollama.

```powershell
docker exec ollama ollama pull imcurie/bge-large-en-v1.5
```

> Note: You can remove the model with the following command: \
> Remove a single model: `docker exec ollama ollama rm imcurie/bge-large-en-v1.5` \
> Remove all ollama data: `docker volume rm ollama # remove all ollama data`

3. Replace `index_config.json` with the following:

```text
[
  {
    "query_index": {
      "type": "HnswEmbeddingIndex",
      "embedding": {
        "type": "OllamaEmbeddingModel",
        "model": "imcurie/bge-large-en-v1.5"
      }
    }
  }
]
```

# Contributing
1. Check the [open issues](https://github.com/synthbot-anon/horsona/issues) for something you can work on. If you're new, check out [good first issues](https://github.com/synthbot-anon/horsona/labels/good%20first%20issue). If you want to work on something that's not listed, post in the thread so we can figure out how to approach it.
2. Post in the thread to claim an issue. You can optionally include your github account in the post so I know whom to assign the issue to.
3. If you're implementing new functionality, create a new folder in `src/horsona/contributions` and `tests/contributions` folders, and put your code there. Make sure to add at least one test case so it's clear how to use your code.
4. Make sure your code works with `poetry run pytest path/to/test/file.py`. If you're modifying something outside of `contributions`, make sure to run the relevant tests.
5. If you have a github account, submit a pull request. Otherwise, post your code somehow in the thread.

# Target feature list
This target feature list is incomplete:
- Video generation (via API) alongside text generation
- Integrations with various text generation interfaces (SillyTavern, Risu, etc.)
- Integration with ComfyUI
- Automated character card adjustments
- Lorebook generation from large text corpora
- Splitting prompts into multiple calls for more reliable generation
- Simultaneously accounting for multiple kinds of data
- In-universe and retrospective consistency checks
- Organizing text corpora into compatible universes
- Support for RPG functionality, like HP, XP, and dice rolls based on a rule book
- Transparent adaptation of video generation prompts to the API & model in use
- Making & rolling back high level edits to character cards
- Continue generating the input, as opposed to responding to it (like non-chat GPTs)
- Jailbreak support
- Fine-tuning dataset creation
- Speech generation and voice morphing the results
- Streaming outputs
- Integration with game engines
- Actions (function call) outputs

If you think of other features you want in a general-purpose chatbot library, let me know in the thread.

# Contact
Post in the PPP.
