FROM python:3.7.2-stretch
RUN apt-get update -y
COPY . /app
WORKDIR /app/python_client/chatbot
RUN pip install redis numpy flask flask-cors
ENTRYPOINT [ "flask" ]
CMD [ "run", "-h", "0.0.0.0" ]