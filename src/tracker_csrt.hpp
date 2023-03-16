#ifndef TRACKER_CSRT
#define TRACKER_CSRT


#include <vector>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include <cufft.h>


/**
 * \brief Class for CSRT tracking algorithm
 */
class TrackerCSRT {
public:

    /**
     * \brief Constructs and initializes tracker
     * 
     * \param[in] frame First frame of the video
     * \param[in] bbox Initial bounding box of the tracked object
     */
    TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox);

    /**
     * \brief Destructor
     */
    ~TrackerCSRT();

    /**
     * \brief Makes one step of algorithm
     * Updates current bounding box according to the transformation of the object in the next frame
     * 
     * \param[in] frame Next frame of the video
     * \param[in] bbox Bounding box of the tracked object in the previous frame
     * \return true - updated successfully, false - tracking failed
     */
    bool update(const cv::Mat& frame, cv::Rect& bbox);

private:

    void buildGaussian(cv::Mat1f& dst);
    
    /**
     * \brief Calculates convolution filter for channel
     * 
     * \param[in] channel_id Channel index
     * \param[in] filter Result destination
     */
    void getChannelFilter(const int channel_id, cv::Mat1f& filter);

    /**
     * \brief Convolves an image with the filter
     * 
     * \param[in] src Input image
     * \param[in] kernel Convolution kernel
     * \param[in] dst Output image of the same size as src
     */
    void convolveCUDA(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst);

    /**
     * \brief Convolves an image with the filter
     * 
     * \param[in] src Input image
     * \param[in] kernel Convolution kernel
     * \param[in] dst Output image of the same size as src
     */
    void convolveOpenCV(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst);
    

    void updateLocation(cv::Rect& bbox);
    void estimateReliabilityMap(cv::Mat& map);
    void updateFilter();


    std::vector<cv::Mat1f> filters;
    std::vector<cv::Mat1f> channels;
    std::vector<float> channelWeights;
    cv::Mat1f idealResponse;

    float* dSrc;
    float* dKernel;
    cufftComplex* dSrcFFT;
    cufftComplex* dKernelFFT;

    static const float filterAdaptationRate;
};


#endif // TRACKER_CSRT