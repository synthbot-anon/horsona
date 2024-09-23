# Horsona: The Swiss Army Knife of Pony Chatbot Creation
Creating a realistic pony chatbot is very difficult. This repo will try to maintain an organized collection of features that a pony chatbot might need in the hopes that future chatbot developers will have a easier time with it.

# Installation
Install the repo to use it:
```bash
pip install --upgrade git+https://github.com/synthbot-anon/horsona
# Create a .env file in your project
curl https://raw.githubusercontent.com/synthbot-anon/horsona/main/.env.example > /path/to/project/.env
```

Or install the repo for development:
```bash
pip install git+https://github.com/synthbot-anon/horsona
git clone git@github.com:synthbot-anon/horsona.git
cd horsona
./dev-install.sh
# Start the dev environment
poetry shell
# Use the .env file in the horsona repo
```

Configure the environment variables in .env:
```bash
vim .env
# Edit .env to include your API keys. Example:
CEREBRAS_API_KEY="csk-xxxxxxxxxxxxxxxxxxxxxxxxx"
FIREWORKS_API_KEY="fw_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

# Running tests
Horsona is a library, not a chatbot application. So there's not a "main" method, and everything is run through tests. Actual chatbots will go into separate repos.
```bash
# To run tests/test_llm.py...

# Edit tests/test_llm.py to use your preferred models. (I'll add in a better way to do this later.)

# Then run the test. (Add -s to the end to show print statements.
poetry run pytest tests/test_llm.py

# Most tests are stochastic since they use external APIs. 
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
- Support continuing prompts, like non-chat GPTs
- Support jailbreaks
- Support the creation of fine-tuning datasets

If you think of other features you want in a general-purpose chatbot library, let me know in the thread.

# Contact
Post in the PPP.
