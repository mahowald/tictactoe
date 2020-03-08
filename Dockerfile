FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

WORKDIR /src/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY setup.py .
COPY tictactoe/ ./tictactoe/
RUN python setup.py install
COPY pytorch_dqn.pt .

ENV TERM=xterm-color

ENTRYPOINT ["python", "-m", "tictactoe"]