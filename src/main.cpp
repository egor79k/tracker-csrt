#include <opencv2/opencv.hpp>
#include "tracker_csrt.hpp"


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./csrt <video_file>" << std::endl;
        return 1;
    }

    cv::VideoCapture video(argv[1]);

    if(!video.isOpened()) {
        std::cout << "Could not open video file" << std::endl; 
        return 1;
    } 

    cv::Mat frame;

    if (!video.read(frame)) {
        std::cout << "Could not read video file" << std::endl; 
        return 1;
    }

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
 
    cv::Rect rect = cv::selectROI(frame, false);
    
    TrackerCSRT tracker(frame, rect);

    cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2); 
    cv::imshow("Tracker", frame);
     
    while(video.read(frame)) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        
        if (tracker.update(frame, rect)) {
            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Tracker", frame);

        if (cv::waitKey(50) == 'q') {
            break;
        }
    }

    return 0;
}