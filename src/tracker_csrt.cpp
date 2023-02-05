#include "tracker_csrt.hpp"


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filter(frame, bbox),
    channels(1),
    channelWeights(1)
{}


void TrackerCSRT::getChannelFilter(const int channel_id, cv::Mat1f& filter) {
    // Calculate the closed-form solution for channel filter using formula (4)
}


// void TrackerCSRT::convolve(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
//     cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_WRAP);
// }


void TrackerCSRT::convolve(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
    // The optimal size for DFT transform
    cv::Size dftSize;

    dftSize.width = cv::getOptimalDFTSize(src.cols + kernel.cols);
    dftSize.height = cv::getOptimalDFTSize(src.rows + kernel.rows);

    cv::Mat1f srcTemp = cv::Mat1f::zeros(dftSize);
    cv::Mat1f kernelTemp = cv::Mat1f::zeros(dftSize);

    // Copy src and kernel to the top-left corners of srcTemp and kernelTemp
    cv::Mat srcRoi(srcTemp, cv::Rect(0, 0, src.cols, src.rows));
    cv::Mat kernelRoi(kernelTemp, cv::Rect(0, 0, kernel.cols, kernel.rows));

    src.copyTo(srcRoi);
    kernel.copyTo(kernelRoi);
    
    // Transform src and kernel into frequency domain
    cv::dft(srcTemp, srcTemp, 0, src.rows);
    cv::dft(kernelTemp, kernelTemp, 0, kernel.rows);
    
    // Multiply matrices elementwise in frequency domain
    cv::mulSpectrums(srcTemp, kernelTemp, srcTemp, 0);
    
    // Transform result from the frequency domain
    cv::idft(srcTemp, srcTemp);
    
    // Copy result to dst
    cv::Rect convRoi(
        kernel.cols,
        kernel.rows,
        cv::abs(src.cols - kernel.cols),
        cv::abs(src.rows - kernel.rows));

    srcTemp(convRoi).copyTo(dst);
}


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    // Temporary use just grayscale channel with 1.0 weight
    cv::Mat1f(frame, bbox).convertTo(channels[0], CV_32F);
    channelWeights[0] = 1.0f;

    cv::Rect wrappedRoi(bbox.width / 2, bbox.height / 2, bbox.width * 2, bbox.height * 2);
    cv::Mat1f resultingConvolution = cv::Mat::zeros(bbox.height, bbox.width, CV_32F);
    cv::Mat1f channelConvolution;

    for (int channel_id = 0; channel_id < channels.size(); ++channel_id) {
        getChannelFilter(channel_id, filter);

        // Replicate image to caclulate cyclic convolution
        cv::Mat1f wrappedChannel(cv::repeat(channels[channel_id], 3, 3), wrappedRoi);

        convolve(wrappedChannel, filter, channelConvolution);

        // Add weighted channel convolution to resulting response
        resultingConvolution = resultingConvolution + channelConvolution * channelWeights[channel_id];
    }

    // Find the location of maximum in convolution response
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(resultingConvolution, &minVal, &maxVal, &minLoc, &maxLoc);

    // Move the bounding box according to maximum
    bbox += (maxLoc - cv::Point(bbox.width / 2, bbox.height / 2));

    return true;
}
