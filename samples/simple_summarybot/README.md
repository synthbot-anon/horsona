# Simple SummaryBot

This project can summarize news, papers, etc. Must have an available text format (e.g. it cannot summarize videos, but it can summarize a video transcript).

## Setup

1. Edit the following configuration files according to the [horsona README](https://github.com/synthbot-anon/horsona/blob/main/README.md):
   - `llm_config.json` - Configure LLM settings
   - `index_config.json` - Configure index settings 
   - `.env` - Copy from `.env.example` and adjust it to set environment variables

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run the application:
   ```bash
   poetry run python -m simple_summarybot
   ```


