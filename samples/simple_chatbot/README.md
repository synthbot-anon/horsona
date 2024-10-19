# horsona/simple_chatbot
This application demos the following functionality:
- Persona-based chat. `persona_config.json` can contain arbitrary data as long it includes a `name` field. This information is passed directly to the LLM when generating responses.
- Short-term memory (recent messages)
- Long-term memory (embeddings database)
- **TODO: support saving & restoring memory.**

# Installation
1. Clone the repo and navigate to the `simple_chatbot/bin/windows` folder. (On Linux: `simple_chatbot/bin/linux`)

```powershell
git clone https://github.com/synthbot-anon/horsona.git
cd horsona/samples/simple_chatbot/bin/windows
# On Linux: cd horsona/samples/simple_chatbot/bin/linux
```

2. Copy `.env.example` into `.env` and edit `.env` with your OpenAI API key.
3. Run `simple_chatbot.bat`. (On Linux: `simple_chatbot.sh`)

> This uses OpenAI for ease of setup. This is VERY SLOW with the OpenAI API.
> For faster responses:
> - Configure `llm_config.json` to use Cerebras or Fireworks. (See below.)\
> - Configure `index_config.json` to use Ollama for embeddings. (See below.)

# Using a different LLM API
1. Edit `.env` with your new LLM's API key.
2. Edit `llm_config.json` to use your new LLM. Supported "types" include:
   - AsyncCerebrasEngine
   - AsyncGroqEngine
   - AsyncFireworksEngine
   - AsyncOpenAIEngine
   - AsyncAnthropicEngine
   - AsyncTogetherEngine

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
