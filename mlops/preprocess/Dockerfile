FROM python:latest
RUN pip install -U scikit-learn 
RUN pip install numpy
COPY ./preprocess.py /mlop1/
WORKDIR /mlop1/
ENTRYPOINT ["python","preprocess.py"]