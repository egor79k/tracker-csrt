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


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filters(3),
    channels(3),
    channelWeights(3)
{
    std::vector<cv::Mat1f> temp(3);
    cv::split(cv::Mat(frame, bbox), temp.data());

    for (int channel_id = 0; channel_id < filters.size(); ++ channel_id) {
        temp[channel_id].convertTo(filters[channel_id], CV_32F);
    }

    size_t realSize = bbox.width * bbox.height * sizeof(float);
    size_t complexSize = bbox.width * bbox.height * sizeof(cufftComplex);

    cudaMalloc(&dSrc, realSize);
    cudaMalloc(&dKernel, realSize);
    cudaMalloc(&dSrcFFT, complexSize);
    cudaMalloc(&dKernelFFT, complexSize);
}


TrackerCSRT::~TrackerCSRT() {
    cudaFree(dSrc);
    cudaFree(dKernel);
    cudaFree(dSrcFFT);
    cudaFree(dKernelFFT);
}


void TrackerCSRT::getChannelFilter(const int channel_id, cv::Mat1f& filter) {
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


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    std::vector<cv::Mat1f> temp(3);
    cv::split(cv::Mat(frame, bbox), temp.data());

    for (int channel_id = 0; channel_id < channels.size(); ++ channel_id) {
        temp[channel_id].convertTo(channels[channel_id], CV_32F);
    }

    channelWeights[0] = 1.0f;
    channelWeights[1] = 1.0f;
    channelWeights[2] = 1.0f;

    cv::Mat1f resultingConvolution = cv::Mat::zeros(bbox.height, bbox.width, CV_32F);
    cv::Mat1f channelConvolution;

    for (int channel_id = 0; channel_id < channels.size(); ++channel_id) {
        // getChannelFilter(channel_id, filter);

        convolveCUDA(channels[channel_id], filters[channel_id], channelConvolution);

        // Add weighted channel convolution to resulting response
        resultingConvolution = resultingConvolution + channelConvolution * channelWeights[channel_id];
    }

    // cv::normalize(resultingConvolution, resultingConvolution, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Conv", resultingConvolution);
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

    return true;
}
