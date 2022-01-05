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

struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);
void detectFace(cv::Mat &frame, int frame_id);
bool fileExists(const std::string &path);
int getch_echo(bool echo);

Mat reconstructFace(const Mat face);
double getSimilarity(const Mat A, const Mat B);

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p);

string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;

static char output_dir[200];
static const char *model_path;
Ptr<FisherFaceRecognizer> model;
const float UNKNOWN_PERSON_THRESHOLD = 0.5f;
static struct timespec start_time;
static struct timespec end_time;
int detectFlag = 0;

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

    cv::Mat frame;

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

    cv::VideoCapture camera(2);

    framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
    std::ofstream ofs("/dev/fb0");

    //* Load model
    model = FisherFaceRecognizer::create();
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

    int x_width = fb_info.xres_virtual;
    int y_height = fb_info.yres_virtual;
    camera.set(cv::CAP_PROP_FRAME_WIDTH, x_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, y_height);
    int pixel_size = fb_info.bits_per_pixel >> 3;
    int fid = 0;
    char c;
    int cur_loc = 0;
    detectFlag = 0;
    while (true)
    {
        camera >> frame;
        cv::Size2f frame_size = frame.size();
        c = getch_echo(true); // detect keyboard input and output to c

        if (c == 'c') // if input is 'c'
        {
            detectFlag = 1;
        }

        if (detectFlag) // start to count time and detect face
        {
            clock_gettime(CLOCK_MONOTONIC, &start_time);
            detectFace(frame, fid);
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2BGR565);
        for (int y = 0; y < frame_size.height; y++) //output the frame to framebuffer
        {
            cur_loc = y * pixel_size * x_width;
            ofs.seekp(cur_loc);

            ofs.write(frame.ptr<char>(y), x_width * pixel_size);
        }

        if (detectFlag) // if input is 'c' then pause the program
        {
            while (true)
            {
            }
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

void detectFace(cv::Mat &frame, int frame_id)
{
    vector<Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (int i = 0; i < faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

        rectangle(frame, faces[i], Scalar(255, 0, 255), 1, 8, 0);
        cv::Mat faceROI = frame_gray(faces[i]);
        cv::Mat res;
        cv::resize(faceROI, res, Size(128, 128), 0, 0, INTER_LINEAR);

        Mat reconstructedFace = reconstructFace(res);

        double similarity = getSimilarity(res, reconstructedFace);

        Point origin;
        origin.x = center.x - faces[i].width / 2;
        origin.y = center.y + faces[i].height / 2 + 10;

        if (similarity < UNKNOWN_PERSON_THRESHOLD)
        {
            int predictedLabel;
            double confidence;
            model->predict(res, predictedLabel, confidence);
            clock_gettime(CLOCK_MONOTONIC, &end_time);
            double timeElapsed = (double)abs(timespecDiff(&end_time, &start_time)) / 1000000.0;
            //printf("end_time sec : %ld end_time nsec : %ld\n", end_time.tv_sec, end_time.tv_nsec);
            //printf("start_time sec : %ld start_time nsec : %ld\n", start_time.tv_sec, start_time.tv_nsec);
            String text;
            if (predictedLabel == 16)
            {
                text = format("310552015");
                putText(frame, text, origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);
                origin.y = origin.y + 35;
                putText(frame, format("%.2lfms", timeElapsed), origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);
            }
            else
            {
                text = format("310552051");
                putText(frame, text, origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);
                origin.y = origin.y + 35;
                putText(frame, format("%.2lfms", timeElapsed), origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);
            }
        }
        else
        {
            putText(frame, format("Unknown"), origin, FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 2, 8, 0);
        }

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
}

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

Mat reconstructFace(const Mat face)
{

    // Since we can only reconstruct the face for some types of FaceRecognizer models (ie: Eigenfaces or Fisherfaces),
    // we should surround the OpenCV calls by a try/catch block so we don't crash for other models.
    try
    {
        // Get some required data from the FaceRecognizer model.
        // Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat eigenvectors = model->getEigenVectors();
        // Mat averageFaceRow = model->get<Mat>("mean");
        Mat averageFaceRow = model->getMean();

        int faceHeight = face.rows;

        // Project the input image onto the PCA subspace.
        Mat projection = LDA::subspaceProject(eigenvectors, averageFaceRow, face.reshape(1, 1));
        //printMatInfo(projection, "projection");

        // Generate the reconstructed face back from the PCA subspace.
        Mat reconstructionRow = LDA::subspaceReconstruct(eigenvectors, averageFaceRow, projection);
        //printMatInfo(reconstructionRow, "reconstructionRow");

        // Convert the float row matrix to a regular 8-bit image. Note that we
        // shouldn't use "getImageFrom1DFloatMat()" because we don't want to normalize
        // the data since it is already at the perfect scale.

        // Make it a rectangular shaped image instead of a single row.
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        // Convert the floating-point pixels to regular 8-bit uchar pixels.
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        //printMatInfo(reconstructedFace, "reconstructedFace");

        return reconstructedFace;
    }
    catch (cv::Exception e)
    {
        //cout << "WARNING: Missing FaceRecognizer properties." << endl;
        return Mat();
    }
}

// Compare two images by getting the L2 error (square-root of sum of squared error).
double getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols)
    {
        // Calculate the L2 relative error between the 2 images.
        double errorL2 = norm(A, B, NORM_L2);
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else
    {
        //cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
        return 100000000.0; // Return a bad value
    }
}

int64_t timespecDiff(struct timespec *timeA_p, struct timespec *timeB_p)
{
    // return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) -
    //        ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
    return (timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec - timeB_p->tv_nsec;
}
