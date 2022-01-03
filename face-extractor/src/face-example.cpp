/**
    Theme: Face Detection
    compiler: Visual Studio 2010 with OpenCV 2.4 beta
    Date: 101/04/28
    Author: HappyMan
    Blog: https://cg2010studio.wordpress.com/
*/
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core.hpp"
 
#include <iostream>
#include <stdio.h>
 
using namespace std;
using namespace cv;
 
/** Function Headers */
void detectAndDisplay( Mat frame );
 
/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
 
/**
 * @function main
 */
int main( int argc, const char** argv )
{
  CvCapture* capture;
  Mat frame;
 
  //-- 1. Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    char* imageName = "test.jpg";
 
    Mat image;
    image = imread( imageName, 1 );
 
    // imshow( imageName, image );
    detectAndDisplay(image);
 
    waitKey(0);
  return 0;
}
 
/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
   std::vector<Rect> faces;
   Mat frame_gray;
 
   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
   equalizeHist( frame_gray, frame_gray );
   //-- Detect faces
   face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(1, 1) );
 
   for( int i = 0; i < faces.size(); i++ )
    {
      Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
      ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 2, 8, 0 );
 
      Mat faceROI = frame_gray( faces[i] );
      std::vector<Rect> eyes;
      imwrite(format("./faces/face_%02d.png", i), faceROI);
      //-- In each face, detect eyes
      eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(.5, .5) );
 
      for( int j = 0; j < eyes.size(); j++ )
       {
         Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
         int radius = cvRound( (eyes[j].width + eyes[i].height)*0.25 );
         circle( frame, center, radius, Scalar( 255, 0, 0 ), 3, 8, 0 );
       }
    }
   //-- Show what you got
   //    imshow( window_name, frame );
   imwrite( "result.jpg", frame );

}