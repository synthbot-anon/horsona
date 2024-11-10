## Poetry
* [Pipx](https://pipx.pypa.io/stable/installation/)
* [Poetry 1.7+](https://python-poetry.org/docs/)
    * [StackOverflow](https://stackoverflow.com/questions/70003829/poetry-installed-but-poetry-command-not-found): you might have to manually set some `$PATH` environment variables

## Python 3.11
* [Python 3.11](https://www.python.org/downloads/)
* [StackOverflow](https://askubuntu.com/questions/1512005/python3-11-install-on-ubuntu-24-04): Python 3.11 may not be available directly through official Ubuntu releases.
* You may have to manually set the poetry environment:
      ```bash
      $ poetry env use /usr/bin/python3.11
      ```

