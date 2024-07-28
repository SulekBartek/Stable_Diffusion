FROM python:3.11.0

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/serving_api

ADD ./serving_api /opt/serving_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/serving_api/requirements.txt

RUN chmod +x /opt/serving_api/run.sh
RUN chown -R api-user:api-user ./

USER api-user

EXPOSE 8001

CMD ["bash", "./run.sh"]
