#include <opencv2/opencv.hpp>
#include "tracker_csrt.hpp"


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./csrt <input_video_file> [output_video_file]" << std::endl;
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

    cv::Rect rect = cv::selectROI(frame, false);
    
    TrackerCSRT tracker(frame, rect);

    cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2); 
    cv::imshow("Tracker", frame);

    cv::VideoWriter video_writer;

    if (argc == 3) {
        video_writer.open(argv[2], cv::VideoWriter::fourcc('M','P','4','V'), 24, frame.size());
    }

    while(video.read(frame)) {
        if (tracker.update(frame, rect)) {
            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Tracker", frame);

        if (argc == 3) {
            video_writer.write(frame);
        }

        if (cv::waitKey(50) == 'q') {
            break;
        }
    }

    if (argc == 3) {
        video_writer.release();
    }

    return 0;
}