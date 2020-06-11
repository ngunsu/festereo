FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN conda install -y click opencv 
RUN conda install -y scipy
RUN conda install -y scikit-image 
RUN python3 -m pip install pytorch-lightning==0.7.6 adabound
RUN python3 -m pip install torchsummary==1.5.1

RUN mkdir -p /root/datasets
