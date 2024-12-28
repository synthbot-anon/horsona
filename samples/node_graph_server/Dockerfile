FROM ubuntu:24.04

RUN apt update --fix-missing
RUN apt install -y \
    sudo \
    git \
    python3 \
    pipx \
    adduser

RUN useradd --create-home --shell /bin/bash --user-group celestia
RUN mkdir /host && chown celestia:celestia /host

USER celestia
RUN pipx ensurepath
RUN pipx install poetry
RUN pipx upgrade poetry

COPY ./samples/node_graph_server /home/celestia/
RUN pipx run poetry install -C /home/celestia/node_graph_server

USER celestia
WORKDIR /host
ENTRYPOINT ["pipx", "run", "poetry", "-C", "/home/celestia/node_graph_server", "run", \
            "python", "-m", "node_graph_server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

