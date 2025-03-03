FROM ubuntu:latest

RUN apt update
RUN apt install -y --no-install-recommends \
  git python3 g++ cmake make ninja-build zlib1g-dev

COPY downloadCompilers.sh ./

RUN chmod +x downloadCompilers.sh
RUN ./downloadCompilers.sh
