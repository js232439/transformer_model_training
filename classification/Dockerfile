FROM python:3.9.6

WORKDIR /sandbox

COPY statements.csv .

COPY few-shot-classification.py .

RUN pip3 install tensorflow transformers requests beautifulsoup4 pandas tf-keras

CMD ["sh", "-c", "python3 few-shot-classification.py"]
