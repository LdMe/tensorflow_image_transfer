Neural Style transfer with tensorflow

----------------------CPU----------------------------------------------------------------------------
To run tensorflow-cpu version, first uncomment the second line of the dockerfile and comment the fourth.
Then you can just run the docker-compose file or build the dockerfile and run it.

To run dockerized version: 

docker-compose up

To make changes in the image:

docker-compose up --build

To run dockerfile:

docker run -v output:/output <docker-image-name>
-----------------------------------------------------------------------------------------------------
---------------------GPU-----------------------------------------------------------------------------

To run tensorflow-gpu version you don't need to touch the dockerfile, but you need to do some installation steps before being able to run the docker image

You can see the instructions in the following link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

after installing and checking that it works:

you have to execute run.sh

you can add -b as first parameter to build the image before it runs

As second parameter, you can specify the output volume where the new images will be saved.

Example:

./run.sh -b /home/myusername/output

You can also change the environment variables from the script:

* CONTENT_PATH => route of the image you want to use as CONTENT, by default it will search for content.jpg in the same directory
* STYLE_PATH => route of the image you want to use as STYLE, by default it will search for style.jpg in the same directory
* CONTENT_LAYER => the layer of the network to take the image from, the inner it is (block2_conv1 > block1_conv2 ), the more it will look like the style.
* STYLE_WEIGHT => the weight (importance) you want the style to have, bigger numbers will create more abstract results
* CONTENT_WEIGHT => the weight for the content, the bigger numbers will give more importance to the content, making the result more similar to the input content file
* STEPS => how many images you want to save, by default they will be saved more or less every 100 iterations, but that can be changed yn the last lines of style_transfer.py

To change the environment variables, you have to go to line 19 in run.sh, and write -e <environment-variable>=<value> after --gpus all
Example of line 19 in run.sh:

docker run -it -v $volume:/output --gpus all -e STEPS=20 -e CONTENT_LAYER="block4_conv2"  -e STYLE_WEIGHT=1e5 style_transfer