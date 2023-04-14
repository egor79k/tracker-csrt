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
    /**
     * \brief Extracts template from frame and splits it into channels
     * 
     * \param[in] frame Current frame of the video
     * \param[in] bbox Bounding box of the tracked object in the current frame
     */
    void updateChannels(const cv::Mat& frame, const cv::Rect& bbox);

    /**
     * \brief Calcultes Gaussian kernel with size of dst
     * 
     * \param[in] dst Output matrix
     */
    void buildGaussian(cv::Mat1f& dst);

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
    
    /**
     * \brief Updates object location in the next frame
     * Estimates a new position of the object
     * as the max in weighted channel correlation response
     * 
     * \param[in] bbox Bounding box of the tracked object to update
     */
    void updateLocation(cv::Rect& bbox);

    /**
     * \brief Builds segmentation mask of the object
     * 
     * \param[in] map Output reliability map
     */
    void estimateReliabilityMap(cv::Mat& map);

    /**
     * \brief Updates channel filters
     */
    void updateFilter();

    void estimateFilter(const int channelId, const cv::Mat1f& relMap, cv::Mat1f& newFilter);


    std::vector<cv::Mat1f> filters;
    std::vector<cv::Mat1f> channels;
    std::vector<float> channelWeights;
    cv::Mat1f idealResponse;

    float* dSrc;
    float* dKernel;
    float* dGaussian;
    float* dRelMap;
    cufftComplex* dSrcFFT;
    cufftComplex* dKernelFFT;
    cufftComplex* dGaussianFFT;
    cufftComplex* dLagrangianFFT;
    cufftComplex* dConv1;
    cufftComplex* dConv2;
    cufftComplex* dTemp1;
    // cufftComplex* dTemp2;

    static const float filterAdaptationRate;
};


#endif // TRACKER_CSRT