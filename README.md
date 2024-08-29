# Horsona: The Swiss Army Knife of Pony Chatbot Creation
Creating a realistic pony chatbot is very difficult. This repo will try to maintain an organized collection of features that a pony chatbot might need in the hopes that future chatbot developers will have a easier time with it.

I don't know yet what functionality will be in here or how it will be organized.
Here's a candidate list:
- Video generation (via API) alongside text generation
- Integrations with various interfaces (SillyTavern, Risu, etc.)
- Automated character card adjustments
- Lorebook generation from large text corpora
- Splitting prompts into multiple calls for more reliable generation
- Simultaneously accounting for multiple kinds of data
- In-universe and retrospective consistency checks
- Organizing text corpora into compatible universes
- Support for RPG functionality, like HP, XP, and dice rolls based on a rule book
- Transparent adaptation of video generation prompts to the API & model in use
- Making & rolling back high level edits to character cards

# Contributing
1. Check the [open issues](https://github.com/synthbot-anon/horsona/issues) for something you can work on. If you want to work on something new, post in the thread so we can sanity check it.
2. Post in the thread to claim an issue. You can optionally include your github account in the post so I know whom to assign the issue to.
3. If you're implementing new functionality, put the code in the `src/horsona/contributions` and `tests/contributions` folders. Make sure to add at least one test case so it's clear how to use your code.
4. Make sure your code works with `poetry run pytest path/to/test/file.py`. If you're modifying something outside of `contributions`, make sure to run the relevant tests.
5. If you have a github account, submit a pull request. Otherwise, post your code somehow in the thread.
