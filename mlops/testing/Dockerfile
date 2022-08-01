FROM python:latest
RUN pip install -U scikit-learn
RUN pip install numpy
COPY ./testing_data.py /mlop3/
WORKDIR /mlop3/
ENTRYPOINT ["python","testing_data.py"]