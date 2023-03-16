#include <cmath>
#include <cufft.h>
#include <cufftXt.h>
#include "tracker_csrt.hpp"


typedef float2 Complex;


// Complex conjugate
static __device__ __host__ inline Complex ComplexConjugate(Complex a) {
  Complex c;
  c.x = a.x;
  c.y = -a.y;
  return c;
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], ComplexConjugate(b[i])), scale);
  }
}


const float TrackerCSRT::filterAdaptationRate = 0.02;


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filters(3),
    channels(3),
    channelWeights(3),
    idealResponse(cv::Mat1f::zeros(bbox.height, bbox.width))
{
    std::vector<cv::Mat1f> temp(3);
    cv::split(cv::Mat(frame, bbox), temp.data());

    for (int channelId = 0; channelId < filters.size(); ++ channelId) {
        temp[channelId].convertTo(filters[channelId], CV_32F);
    }

    size_t realSize = bbox.width * bbox.height * sizeof(float);
    size_t complexSize = bbox.width * bbox.height * sizeof(cufftComplex);

    cudaMalloc(&dSrc, realSize);
    cudaMalloc(&dKernel, realSize);
    cudaMalloc(&dSrcFFT, complexSize);
    cudaMalloc(&dKernelFFT, complexSize);

    buildGaussian(idealResponse);
}


TrackerCSRT::~TrackerCSRT() {
    cudaFree(dSrc);
    cudaFree(dKernel);
    cudaFree(dSrcFFT);
    cudaFree(dKernelFFT);
}


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    std::vector<cv::Mat1f> temp(3);
    cv::split(cv::Mat(frame, bbox), temp.data());

    for (int channelId = 0; channelId < channels.size(); ++ channelId) {
        temp[channelId].convertTo(channels[channelId], CV_32F);
    }

    channelWeights[0] = 1.0f;
    channelWeights[1] = 1.0f;
    channelWeights[2] = 1.0f;

    updateLocation(bbox);
    
    updateFilter();

    return true;
}


void TrackerCSRT::buildGaussian(cv::Mat1f& dst) {
    cv::Size size = dst.size();
    dst = cv::Mat1f::zeros(size);
    const float xSigma = size.width / 3;
    const float ySigma = size.height / 3;
    const float xFactor = 1 / (2 * xSigma * xSigma);
    const float yFactor = 1 / (2 * ySigma * ySigma);
    float sum = 0;

    for (int x = 0; x < size.width; ++x) {
        for (int y = 0; y < size.height; ++y) {
            int xc = x - size.width / 2;
            int yc = y - size.height / 2;

            dst.at<float>(y, x) = std::exp(-(xc * xc * xFactor + yc * yc * yFactor));
            sum += dst.at<float>(y, x);
        }
    }

    dst /= sum;

    // cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Gaussian", dst);
    // cv::waitKey(0);
}


void TrackerCSRT::getChannelFilter(const int channelId, cv::Mat1f& filter) {
    // Calculate the closed-form solution for channel filter using formula (4)
}


// void TrackerCSRT::convolve(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
//     cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_WRAP);
// }


void TrackerCSRT::convolveCUDA(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
    // The optimal size for DFT transform
    size_t width = src.cols;
    size_t height = src.rows;

    cv::Size dftSize;
    dftSize.width = width;
    dftSize.height = height;

    cv::Mat1f srcTemp = cv::Mat1f::zeros(dftSize);
    cv::Mat1f kernelTemp = cv::Mat1f::zeros(dftSize);

    // Copy src and kernel to the top-left corners of srcTemp and kernelTemp
    cv::Mat srcRoi(srcTemp, cv::Rect(0, 0, src.cols, src.rows));
    cv::Mat kernelRoi(kernelTemp, cv::Rect(0, 0, kernel.cols, kernel.rows));

    src.copyTo(srcRoi);
    kernel.copyTo(kernelRoi);

    // Copy src and kernel to device
    cudaMemcpy2D(dSrc, width * sizeof(float), srcTemp.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dKernel, width * sizeof(float), kernelTemp.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Apply FFT to srcTemp and kernel
    cufftHandle plan1;
    cufftPlan2d(&plan1, height, width, CUFFT_R2C);
    cufftExecR2C(plan1, dSrc, dSrcFFT);
    cufftExecR2C(plan1, dKernel, dKernelFFT);
    cufftDestroy(plan1);

    // Multiply elementwise srcTemp and kernel in Fourier domain
    ComplexPointwiseMulAndScale<<<32, 256>>>(dSrcFFT, dKernelFFT, width * height, 1.0f / (width * height));

    // Apply IFFT to srcTemp
    cufftHandle plan2;
    cufftPlan2d(&plan2, height, width, CUFFT_C2R);
    cufftExecC2R(plan2, dSrcFFT, dSrc);
    cufftDestroy(plan2);

    // Copy result to host
    dst = cv::Mat1f::zeros(height, width);
    cudaMemcpy2D(dst.data, width * sizeof(float), dSrc, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);

    // cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Copy", dst);
    // cv::waitKey(0);
}


void TrackerCSRT::convolveOpenCV(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
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


void TrackerCSRT::updateLocation(cv::Rect& bbox) {
    cv::Mat1f resultingConvolution = cv::Mat::zeros(bbox.height, bbox.width, CV_32F);
    cv::Mat1f channelConvolution;

    for (int channelId = 0; channelId < channels.size(); ++channelId) {
        // getChannelFilter(channelId, filter);

        convolveCUDA(channels[channelId], filters[channelId], channelConvolution);

        // Add weighted channel convolution to resulting response
        resultingConvolution = resultingConvolution + channelConvolution * channelWeights[channelId];

        filters[channelId] = 0.98 * filters[channelId] + 0.02 * channels[channelId];
    }

    // Show correlation response in the frame corner
    // cv::Mat rgbConv;
    // cv::normalize(resultingConvolution, resultingConvolution, 0, 255, cv::NORM_MINMAX);
    // cv::cvtColor(resultingConvolution, rgbConv, cv::COLOR_GRAY2BGR);
    // cv::Mat frameRoi(frame, cv::Rect(0, 0, rgbConv.cols, rgbConv.rows));
    // rgbConv.copyTo(frame(cv::Rect(0, 0, rgbConv.cols, rgbConv.rows)));
    // cv::imshow("Conv", frame);
    // cv::waitKey(0);

    // Find the location of maximum in convolution response
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(resultingConvolution, &minVal, &maxVal, &minLoc, &maxLoc);

    // Move the bounding box according to maximum
    // bbox += (maxLoc - cv::Point(bbox.width / 2, bbox.height / 2));
    bbox += (maxLoc - cv::Point(bbox.width * static_cast<int>(2 * maxLoc.x / bbox.width), bbox.height * static_cast<int>(2 * maxLoc.y / bbox.height)));
}


void TrackerCSRT::estimateReliabilityMap(cv::Mat& map) {
    // Temporary use constant mask
    cv::normalize(idealResponse, map, 0, 1, cv::NORM_MINMAX);
    cv::threshold(map, map, 0.5, 1, cv::THRESH_BINARY);
}


void TrackerCSRT::updateFilter() {
    cv::Mat1f maskedFilter;
    cv::Mat1f constrainedFilter;
    cv::Mat1f temp1;
    cv::Mat1f temp2;

    cv::Mat relMap;
    estimateReliabilityMap(relMap);

    int mu = 5;
    const int beta = 3;
    const float lambda = 0.01;
    const int D = relMap.rows * relMap.cols;
    const float denomIncrement = lambda / (2 * D);

    for (int channelId = 0; channelId < channels.size(); ++channelId) {
        cv::Mat1f filter = filters[channelId].clone();
        cv::Mat1f trainROI = channels[channelId];
        cv::Mat1f L = cv::Mat1f::zeros(filter.rows, filter.cols);

        int iter = 1;

        while (iter--) {
            convolveCUDA(trainROI, idealResponse, temp1);
            convolveCUDA(trainROI, trainROI, temp2);

            cv::multiply(filter, relMap, maskedFilter);
            cv::divide(temp1 + (filterAdaptationRate * maskedFilter - L), temp2 + mu, constrainedFilter);
            
            cv::idft(L + mu * constrainedFilter, temp1, 0, filter.rows);
            cv::multiply(relMap, temp1, temp2);
            filter = temp2 / (denomIncrement + mu);

            L = L + filterAdaptationRate * (constrainedFilter - filter);

            mu *= beta;
        }

        filters[channelId] = (1 - filterAdaptationRate) * filters[channelId] + filterAdaptationRate * filter;
    }
}
