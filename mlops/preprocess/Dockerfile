FROM python:3.7-slim
RUN pip install -U scikit-learn numpy
COPY ./preprocess.py /mlop1/
WORKDIR /mlop1/
ENTRYPOINT ["python","preprocess.py"]