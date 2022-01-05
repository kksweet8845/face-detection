
# reco:
# 	g++ -g -Wall ./src/face-recognize.cpp -o fc -I/tools/opencv-3.4.7/include/opencv4 -L/tools/opencv-3.4.7/lib -lopencv_world

# performance:
# 	g++ -g -Wall ./src/performance.cpp -o perf -I/tools/opencv-3.4.7/include/opencv4 -L/tools/opencv-3.4.7/lib -lopencv_world

performance:
	arm-linux-gnueabihf-g++ ./src/performance.cpp -o perf \
-I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
-I /usr/local/arm-opencv2/install/include/ -L /usr/local/arm-opencv2/install/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
-lpthread -lopencv_world

reco:
	arm-linux-gnueabihf-g++ ./src/face-recognize.cpp -o fc \
-I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
-I /usr/local/arm-opencv2/install/include/ -L /usr/local/arm-opencv2/install/lib/ \
-Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
-lpthread -lopencv_world