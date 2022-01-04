
# opencv:
# 	g++ -g -Wall ./src/face-recognize.cpp -o fc -I/tools/opencv-3.4.7/include -L/tools/opencv-3.4.7/lib -lopencv_world

all:
	arm-linux-gnueabihf-g++ ./src/face-recognize.cpp -o fc \
-I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
-I /usr/local/arm-opencv2/install/include/ -L /usr/local/arm-opencv2/install/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
-lpthread -lopencv_world
