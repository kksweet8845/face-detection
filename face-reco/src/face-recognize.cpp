#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/face.hpp"
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
#include <vector>
#include <pthread.h>

using namespace std;
using namespace cv;
using namespace cv::face;

struct framebuffer_info
{
    uint32_t bits_per_pixel; // depth of framebuffer
    uint32_t xres_virtual;   // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
    uint32_t xres; // visible pixel in a row
};

struct arg_struct
{
    int h_start;
    int h_end;
} h_args;

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);
void detectFace(cv::Mat &frame, int frame_id);
void *Pthread_fun(void *arguments);
void *Thread_detect(void *arg);

bool fileExists(const std::string &path)
{
    bool ret = false;
    if ((access(path.c_str(), F_OK)) != -1)
    {
        ret = true;
    }
    return ret;
}

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
static const char *model_path;
Ptr<EigenFaceRecognizer> model;
int pixel_size = 0;
int x_width = 0;
Mat frame;
vector<Rect> faces;

int main(int argc, const char *argv[])
{

    if (argc < 2)
    {
        printf("Usage ./fc <model.bin>\n");
        return 0;
    }

    model_path = argv[1];

    if (!fileExists(model_path))
    {
        printf("The model file is not existed\n");
        exit(2);
    }

    fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);

    // Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        printf("loading face error\n");
        return -1;
    };

    if (!eyes_cascade.load(eyes_cascade_name))
    {
        printf("loading eyes error\n");
        return -1;
    }

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");

    std::ofstream ofs("/dev/fb0");
    cv::VideoCapture camera(2);

    //* Load model
    model = EigenFaceRecognizer::create();
    model->read(model_path);

    if (!camera.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    time_t rawtime;
    struct tm *timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Current local time and date: %s\n", asctime(timeinfo));

    int fid = 0;
    x_width = fb_info.xres_virtual;
    int y_height = fb_info.yres_virtual;
    camera.set(cv::CAP_PROP_FRAME_WIDTH, x_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, y_height);
    pixel_size = fb_info.bits_per_pixel >> 3;
    char c;
    bool isStart = false;

    while (true)
    {
        camera >> frame;

        c = getch_echo(true);

        cv::Size2f frame_size = frame.size();

        if (c == 'c')
        {
            isStart = true;
            printf("type c\n");
        }

        if (isStart) // start to detect face
            detectFace(frame, fid);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);
        // int cur_loc = 0;
        // for (int y = 0; y < frame_size.height; y++)
        // {
        //     cur_loc = y * pixel_size * x_width;
        //     ofs.seekp(cur_loc);
        //     ofs.write(frame.ptr<char>(y), x_width * pixel_size);
        // }
        int frame_seg = frame_size.height / 3;
        struct arg_struct h1_args = {0, frame_seg};
        struct arg_struct h2_args = {frame_seg - 1, frame_seg * 2};
        struct arg_struct h3_args = {frame_seg * 2 - 1, frame_size.height};
        pthread_t thread_pool[3];
        if (pthread_create(&thread_pool[0], NULL, Pthread_fun, (void *)&h1_args) != 0)
        {
            printf("error thread : %d\n", 0);
        }
        if (pthread_create(&thread_pool[1], NULL, Pthread_fun, (void *)&h2_args) != 0)
        {
            printf("error thread : %d\n", 1);
        }
        if (pthread_create(&thread_pool[2], NULL, Pthread_fun, (void *)&h3_args) != 0)
        {
            printf("error thread : %d\n", 2);
        }

        for (int i = 0; i < 3; i++)
        {
            pthread_join(thread_pool[i], NULL);
        }
        fid++;
    }

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

cv::Mat frame_gray;
void detectFace(cv::Mat &frame, int frame_id)
{
    int height = 250;
    int width = 250;
    cv::Scalar value(255, 255, 255);
    cv::cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    int count = faces.size();
    pthread_t *thread_pool;
    thread_pool = (pthread_t *)malloc(count * sizeof(pthread_t));

    for (int i = 0; i < count; i++)
    {
        if (pthread_create(&thread_pool[i], NULL, Thread_detect, (void *)&i) != 0)
        {
            printf("error thread : %d\n", i);
        }
    }
    for (int i = 0; i < count; i++)
    {
        pthread_join(thread_pool[i], NULL);
    }
    faces.clear();
    frame_gray = NULL;
}

void *Pthread_fun(void *arguments)
{
    struct arg_struct *args = (struct arg_struct *)arguments;
    int height_start = args->h_start;
    int height_end = args->h_end;
    int cur_loc = 0;
    std::ofstream ofs("/dev/fb0");
    for (int y = height_start; y < height_end; y++)
    {
        cur_loc = y * pixel_size * x_width;
        ofs.seekp(cur_loc);

        ofs.write(frame.ptr<char>(y), x_width * pixel_size);
    }
    return NULL;
}

void *Thread_detect(void *arg)
{
    int i = *(int *)arg;
    for (int i = 0; i < faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

        rectangle(frame, faces[i], Scalar(255, 0, 255), 1, 8, 0);
        cv::Mat faceROI = frame_gray(faces[i]);
        cv::Mat res;
        cv::resize(faceROI, res, Size(128, 128), 0, 0, INTER_LINEAR);

        int predictedLabel;
        double confidence;
        model->predict(res, predictedLabel, confidence);
        Point origin;
        origin.x = center.x;
        origin.y = center.y + faces[i].height / 2;
        String text = format("%d", predictedLabel);
        putText(frame, text, origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);

        printf("Confidence: %f, Label: %d\n", confidence, predictedLabel);

        // imwrite(format("./%s/subject%d_%d.png", output_dir, subjectid, frame_id), faceROI);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (int j = 0; j < eyes.size(); j++)
        {
            eyes[j].x = faces[i].x + eyes[j].x;
            eyes[j].y = faces[i].y + eyes[j].y;
            rectangle(frame, eyes[j], Scalar(255, 0, 0), 3, 8, 0);
        }
    }
    return NULL;
}