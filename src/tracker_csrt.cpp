#include "tracker_csrt.hpp"


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filter(frame, bbox)
{}


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    cv::Mat1f object;
    cv::Mat1f convolution;

    cv::Mat1f(frame, bbox).convertTo(object, CV_32F);

    cv::filter2D(object, convolution, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_WRAP);

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(convolution, &minVal, &maxVal, &minLoc, &maxLoc);

    bbox += (maxLoc - cv::Point(bbox.width / 2, bbox.height / 2));

    return true;
}
