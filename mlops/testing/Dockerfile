FROM python:3.7-slim
RUN pip install -U scikit-learn numpy
COPY ./testing_data.py /mlop3/
WORKDIR /mlop3/
ENTRYPOINT ["python","testing_data.py"]