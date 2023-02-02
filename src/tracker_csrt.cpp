#include "tracker_csrt.hpp"


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filter(frame, bbox),
    channels(1),
    channel_weights(1)
{}


void TrackerCSRT::getChannelFilter(const int channel_id, cv::Mat1f& filter) {
    // Calculate the closed-form solution for channel filter using formula (4)
}


void TrackerCSRT::convolution(const cv::Mat1f& src, const cv::Mat1f& filter, cv::Mat1f& dst) {
    cv::filter2D(src, dst, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_WRAP);
}


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    // Temporary use just grayscale channel with 1.0 weight
    cv::Mat1f(frame, bbox).convertTo(channels[0], CV_32F);
    channel_weights[0] = 1.0f;

    cv::Mat1f resulting_convolution = cv::Mat::zeros(bbox.height, bbox.width, CV_32F);
    cv::Mat1f channel_convolution;

    for (int channel_id = 0; channel_id < channels.size(); ++channel_id) {
        getChannelFilter(channel_id, filter);
        convolution(channels[channel_id], filter, channel_convolution);
        resulting_convolution = resulting_convolution + channel_convolution * channel_weights[channel_id];
    }

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(resulting_convolution, &minVal, &maxVal, &minLoc, &maxLoc);

    bbox += (maxLoc - cv::Point(bbox.width / 2, bbox.height / 2));

    return true;
}
