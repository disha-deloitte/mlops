FROM python:latest
RUN pip install -U scikit-learn
COPY ./preprocess.py /mlop1/
WORKDIR /mlop1/
ENTRYPOINT ["python","preprocess.py"]