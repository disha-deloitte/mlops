FROM python:3.7-slim
RUN pip install -U scikit-learn numpy
COPY ./training_data.py /mlop2/
WORKDIR /mlop2/
ENTRYPOINT ["python","training_data.py"]