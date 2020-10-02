#! /bin/bash

#sudo docker-compose build
volume="$HOME/output"
if [ $# -ge 1 ];
then
	if [ $1 == "-b" ];
	then
		docker build . -t style_transfer
		if [ $# -ge 2 ];
		then
			volume=$2
		fi
	else
		volume=$1
	fi
	
fi
docker run -it -v $volume:/output --gpus all   style_transfer