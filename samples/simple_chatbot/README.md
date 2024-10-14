# horsona/simple_chatbot
This application demos the following functionality:
- Persona-based chat. `persona_config.json` can contain arbitrary data as long it includes a `name` field. This information is passed directly to the LLM when generating responses.
- Short-term memory (recent messages)
- Long-term memory (embeddings database)
- **TODO: support saving & restoring memory.**

# Installation on Linux
1. Run Ollama.

```bash
docker run -d --rm -e OLLAMA_HOST=0.0.0.0:11434 -v "ollama:/root/.ollama" -p 11434:11434 --name ollama ollama/ollama

# You can stop Ollama with the following command:
#   docker container kill ollama # preferred
#   docker container rm -f ollama # alternative
```

2. Load an embedding model into Ollama.

```powershell
docker exec ollama ollama pull imcurie/bge-large-en-v1.5

# You can remove the model with the following command:
#   docker exec ollama ollama rm imcurie/bge-large-en-v1.5 # remove single model)
#   docker volume rm ollama # remove all ollama data

```
3. Clone the repo and navigate to the `simple_chatbot/bin/windows` folder.

```powershell
git clone https://github.com/synthbot-anon/horsona.git
cd horsona/samples/simple_chatbot/bin/linux
```

4. Copy `.env.example` into `.env` and edit `.env` with your API keys.
5. Edit `llm_config.json` based on whichver LLM APIs you have access to. Make sure the config file contains a definition for `reasoning_llm`, which the default does.
6. Edit `index_config.json` based on whichever embedding model you pulled into Ollama. If you copy/pasted the above command to load an embedding model, you don't need to make any changes here.
5. Run `simple_chatbot.sh`.

# Installation on Windows
1. Run Ollama.

```powershell
docker run -d --rm -e OLLAMA_HOST=0.0.0.0:11434 -v "ollama:/root/.ollama" -p 11434:11434  --name ollama ollama/ollama

# You can stop Ollama with the following command:
#   docker container kill ollama # preferred
#   docker container rm -f ollama # alternative
```

2. Load an embedding model into Ollama.

```powershell
docker exec ollama ollama pull imcurie/bge-large-en-v1.5

# You can remove the model with the following command:
#   docker exec ollama ollama rm imcurie/bge-large-en-v1.5 # remove single model)
#   docker volume rm ollama # remove all ollama data

```
3. Clone the repo and navigate to the `simple_chatbot/bin/windows` folder.

```powershell
git clone https://github.com/synthbot-anon/horsona.git
cd horsona/samples/simple_chatbot/bin/windows
```

4. Copy `.env.example` into `.env` and edit `.env` with your API keys.
5. Edit `llm_config.json` based on whichver LLM APIs you have access to. Make sure the config file contains a definition for `reasoning_llm`, which the default does.
6. Edit `index_config.json` based on whichever embedding model you pulled into Ollama. If you copy/pasted the above command to load an embedding model, you don't need to make any changes here.
5. Run `simple_chatbot.bat`.