#ifndef TRACKER_CSRT
#define TRACKER_CSRT


#include <opencv2/opencv.hpp>


class TrackerCSRT {
public:
    TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox);
    bool update(const cv::Mat& frame, cv::Rect& bbox);

private:
    cv::Mat1f filter;
};


#endif // TRACKER_CSRT