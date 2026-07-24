# docker build -t ghcr.io/trec-auto-judge/trec-auto-judge-base:dev-0.0.1 . -f .devcontainer/Dockerfile
FROM ubuntu:jammy

RUN apt-get update \
	&& apt-get install -y git python3.11 python3-pip 

RUN python3.11 -m pip install uv 

ADD pyproject.toml /auto-judge/

WORKDIR /auto-judge

RUN uv venv --python 3.11 /venv \
	&& . /venv/bin/activate \
	&& uv pip install -e .[all]

ENV PATH=/venv/bin/:${PATH}
ENV PYTHONPATH=/auto-judge

RUN python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("punkt");'
