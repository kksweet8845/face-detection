#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <errno.h>
#include <vector>
#include <unistd.h>
#include <cstdio>
#include <termios.h>
#include <time.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

struct framebuffer_info
{
    uint32_t bits_per_pixel; // depth of framebuffer
    uint32_t xres_virtual;   // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
    uint32_t xres; // visible pixel in a row
};

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);
void detectFace(cv::Mat &frame, int frame_id);

int getch_echo(bool echo = true)
{
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~ICANON;
    if (echo)
        newt.c_lflag &= ECHO;
    else
        newt.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

// /usr/local/arm-opencv/install/share/OpenCV/haarcascades
string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

static char output_dir[200];

int main(int argc, const char *argv[])
{

    fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
    // variable to store the frame get from video stream
    cv::Mat frame;

    // Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        printf("loading face error\n");
        return -1;
    };

    if(!eyes_cascade.load(eyes_cascade_name)) {
        printf("loading eyes error\n");
        return -1;
    }
    
    // open video stream device
    cv::VideoCapture camera(0);


    if (!camera.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    // int x_width = fb_info.xres_virtual, y_height = fb_info.yres_virtual;
    // camera.set(cv::CAP_PROP_FRAME_WIDTH, x_width);
    // camera.set(cv::CAP_PROP_FRAME_HEIGHT, y_height);

    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s\n", asctime(timeinfo));

   
    sprintf(output_dir, "d%d_%d_%d_t%d_%d_%d", timeinfo->tm_mday, timeinfo->tm_mon+1, timeinfo->tm_year + 1900,
                                               timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
    mkdir(output_dir, 0777);
    int fid=0;
    while (true)
    {
        camera >> frame;

        // c = getchar();
        // c = getch();
        // c = cv::waitKey(1) & 0xff;
        // c = getch_echo(true);
        // // fflush(stdin);
        // if (c == 'c')
        // {
        //     memset(buf, '\0', 100);
        //     // sprintf(buf, "./img_%d.png", pi++);
        //     // cv::imwrite(buf, frame);
        //     sprintf(buf, "cat /dev/fb0 > img_%d.png", pi++);
        //     if (system(buf) < 0)
        //     {
        //         perror("System call:");
        //     }
        // }

        cv::Size2f frame_size = frame.size();
        // std::cout << frame_size << std::endl;
        detectFace(frame, fid);
        imshow("mointor", frame);
        if (waitKey(25) == 'q') {
            break;
        }
        fid++;
    }

    // closing video stream
    // https://docs.opencv.org/3.4.7/d8/dfe/classcv_1_1VideoCapture.html#afb4ab689e553ba2c8f0fec41b9344ae6
    camera.release();

    return 0;
}

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;      // Used to return the required attrs.
    struct fb_var_screeninfo screen_info; // Used to get attributes of the device from OS kernel.

    int fp = open(framebuffer_device_path, O_RDWR);

    if (fp < 0)
    {
        fprintf(stderr, "Error: Can not open framebuffer device.\n");
    }

    if (ioctl(fp, FBIOGET_VSCREENINFO, &screen_info) == -1)
    {
        fprintf(stderr, "ioctl err: %s.\n", strerror(errno));
    }

    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    fb_info.xres_virtual = screen_info.xres;
    fb_info.yres_virtual = screen_info.yres;
    fb_info.xres = screen_info.xres;

    return fb_info;
}

void detectFace(cv::Mat &frame, int frame_id)
{
    vector<Rect> faces;
    cv::Mat frame_gray;
    //Mat frame2 = frame.clone();
    cv::cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
    // cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);

    for (int i = 0; i < faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width, faces[i].height), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);

        cv::Mat faceROI = frame_gray(faces[i]);
        imwrite(format("./%s/frame_%02d_face_%02d.png", output_dir, frame_id, i), faceROI);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (int j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 3, 8, 0);
        }
    }
}