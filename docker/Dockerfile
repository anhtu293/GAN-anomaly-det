FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN apt update 
RUN apt install -y git libsm6 libxrender1 libgl1-mesa-glx wget zip unzip

RUN pip install opencv-python==4.2.0.32 tqdm

RUN pip install tqdm sklearn matplotlib
