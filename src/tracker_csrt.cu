// #include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
// #include <helper_cuda.h>
// #include <helper_functions.h>
#include "tracker_csrt.hpp"


TrackerCSRT::TrackerCSRT(const cv::Mat& frame, const cv::Rect& bbox) :
    filter(frame, bbox),
    channels(1),
    channelWeights(1)
{
    cv::Mat1f img;
    frame.convertTo(img, CV_32F);
    cv::Mat1f kernel = cv::Mat1f::ones(3, 3);
    cv::Mat1f result;
    convolve(frame, kernel, result);
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Conv", result);
    cv::waitKey(0);
    convolveCUDA(frame, kernel, result);
    // cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Conv", result);
    // cv::waitKey(0);
}


void TrackerCSRT::getChannelFilter(const int channel_id, cv::Mat1f& filter) {
    // Calculate the closed-form solution for channel filter using formula (4)
}


// void TrackerCSRT::convolve(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
//     cv::filter2D(src, dst, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_WRAP);
// }

typedef float2 Complex;
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
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}


void TrackerCSRT::convolveCUDA(const cv::Mat1f& src, const cv::Mat1f& kernel, cv::Mat1f& dst) {
    // The optimal size for DFT transform
    size_t width = src.cols;
    size_t height = src.rows;
    cv::Mat1f kernelTemp = cv::Mat1f::zeros(height, width);

    // Copy src and kernel to the top-left corners of src and kernelTemp
    cv::Mat kernelRoi(kernelTemp, cv::Rect(0, 0, kernel.cols, kernel.rows));
    kernel.copyTo(kernelRoi);


    // cudaMalloc(&dSrc, width * height * sizeof(float));
    // cudaMemcpy(dSrc, src.data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(hSrc.data, dSrc, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy src to device
    float* dSrc;
    size_t dSrcPitch;
    cudaMallocPitch(&dSrc, &dSrcPitch, width * sizeof(float), height);
    cudaMemcpy2D(dSrc, dSrcPitch, src.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    
    // Copy kernel to device
    float* dKernel;
    size_t dKernelPitch;
    cudaMallocPitch(&dKernel, &dKernelPitch, width * sizeof(float), height);
    cudaMemcpy2D(dKernel, dKernelPitch, src.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Allocate complex mat on device for FFT of src
    cufftComplex* dSrcFFT;
    size_t dSrcFFTPitch;
    cudaMallocPitch(&dSrcFFT, &dSrcFFTPitch, width * sizeof(cufftComplex), height);

    // Allocate complex mat on device for FFT of kernel
    cufftComplex* dKernelFFT;
    size_t dKernelFFTPitch;
    cudaMallocPitch(&dKernelFFT, &dKernelFFTPitch, width * sizeof(cufftComplex), height);

    // Apply FFT to src and kernel
    cufftHandle plan1;
    cufftPlan2d(&plan1, width, height, CUFFT_R2C);
    cufftExecR2C(plan1, dSrc, dSrcFFT);
    cufftExecR2C(plan1, dKernel, dKernelFFT);
    cufftDestroy(plan1);

    // Multiply elementwise src and kernel in Fourier domain
    ComplexPointwiseMulAndScale<<<32, 256>>>(dSrcFFT, dKernelFFT, width * height, 1.0f / (width * height));

    // Apply IFFT to src
    cufftHandle plan2;
    cufftPlan2d(&plan2, width, height, CUFFT_C2R);
    cufftExecC2R(plan2, dSrcFFT, dSrc);
    cufftDestroy(plan2);

    cv::Mat1f hSrc = cv::Mat1f::zeros(height, width);
    cudaMemcpy2D(hSrc.data, width * sizeof(float), dSrc, dSrcPitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);

    cudaFree(dSrc);
    cudaFree(dKernel);
    cudaFree(dSrcFFT);
    cudaFree(dKernelFFT);

    cv::normalize(hSrc, hSrc, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Copy", hSrc);
    cv::waitKey(0);

    
    // // Transform src and kernel into frequency domain
    // cv::dft(src, src, 0, src.rows);
    // cv::dft(kernelTemp, kernelTemp, 0, kernel.rows);
    
    // // Multiply matrices elementwise in frequency domain
    // cv::mulSpectrums(src, kernelTemp, src, 0);
    
    // // Transform result from the frequency domain
    // cv::idft(src, src);
    
    // // Copy result to dst
    // cv::Rect convRoi(
    //     kernel.cols,
    //     kernel.rows,
    //     cv::abs(src.cols - kernel.cols),
    //     cv::abs(src.rows - kernel.rows));

    // src(convRoi).copyTo(dst);
}


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
    // cv::Mat1f(frame, bbox).convertTo(filter, CV_32F);

    return true;
}
