#CPU
FROM tensorflow/tensorflow:1.15.4-py3
#GPU
#FROM tensorflow/tensorflow:1.15.4-gpu-py3

#unnecessary
#RUN apt install -y libgl1-mesa-glx

RUN pip install imageio 
RUN pip install scipy
RUN pip install Pillow
RUN pip install opencv-python

ADD . .

CMD python style_transfer.py