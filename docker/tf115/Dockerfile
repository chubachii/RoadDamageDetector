# https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_22-03.html#rel_22-03
# Ubuntu : 20.04
# CUDA   : 11.6.1
# cuDNN  : 8.3.3.40
# Python : 3.8
# Tensorflow : 1.15.5

FROM nvcr.io/nvidia/tensorflow:22.03-tf1-py3
#FROM floydhub/tensorflow:1.4.1-gpu.cuda8cudnn6-py2_aws.19
#FROM wonkyunglee/tensorflow:1.8.0-gpu-py3
#FROM tensorflow/tensorflow:1.4.1-py3

# 環境変数
# ENV http_proxy 'http://id:pass@proxy:port' など

# コマンド実行
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-tk -y

# dockerfiles/requirements.txt をコピー
COPY ../requirements.txt .
RUN pip install -r requirements.txt