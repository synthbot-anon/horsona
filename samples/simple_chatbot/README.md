# horsona/simple_chatbot
This application demos the following functionality:
- Persona-based chat. `persona_config.json` can contain arbitrary data as long it includes a `name` field. This information is passed directly to the LLM when generating responses.
- Short-term memory (recent messages)
- Long-term memory (embeddings database)
- **TODO: support saving & restoring memory.**

# Usage
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
> - Configure `llm_config.json` to use Cerebras or Fireworks.
> - Configure `index_config.json` to use Ollama for embeddings.
