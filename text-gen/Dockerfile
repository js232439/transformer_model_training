FROM python:3.9.6

COPY . /text-gen

WORKDIR /text-gen

RUN pip3 install transformers pandas datasets torch huggingface_hub

CMD ["python3", "text-gen.py"]