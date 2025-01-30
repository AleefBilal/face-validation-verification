FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --extra-index-url https://download.pytorch.org/whl/cpu

#RUN mkdir -p models

#RUN curl -L https://huggingface.co/Aleef/face-validation/resolve/main/version-RFB-640.onnx -o models/version-RFB-640.onnx


RUN pip install --no-cache-dir awslambdaric


COPY . .
COPY .env .

RUN python model_download.py


ENTRYPOINT [ "/usr/local/bin/python3.11", "-m", "awslambdaric" ]
CMD [ "handler.handle_verification" ]