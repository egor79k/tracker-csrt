#include <cmath>
#include <cufft.h>
#include <cufftXt.h>
#include "tracker_csrt.hpp"


typedef float2 Complex;


// Complex conjugate
static __device__ __host__ inline Complex ComplexConjugate(const Complex a) {
  Complex c;
  c.x = a.x;
  c.y = -a.y;
  return c;
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(const Complex a, const Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex substraction
static __device__ __host__ inline Complex ComplexSub(const Complex a, const Complex b) {
  Complex c;
  c.x = a.x - b.x;
  c.y = a.y - b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(const Complex a, const float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex division
static __device__ __host__ inline Complex ComplexDiv(const Complex a, const Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(const Complex a, const Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}

// Complex pointwise multiplication with conjugating and scaling
static __global__ void ComplexPointwiseMulConjScale(const Complex* a, const Complex* b, Complex* c, int size, float scale) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexScale(ComplexMul(a[i], ComplexConjugate(b[i])), scale);
    }
}

// Complex pointwise scaling
static __global__ void ComplexPointwiseDiv(const Complex* a, const Complex* b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexDiv(a[i], b[i]);
    }
}

// Complex pointwise scaling
static __global__ void ComplexPointwiseScale(const Complex* a, const float* b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexScale(a[i], b[i]);
    }
}

// Complex pointwise adding
static __global__ void ComplexPointwiseAdd(const Complex* a, const Complex* b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexAdd(a[i], b[i]);
    }
}

// Complex pointwise substracting
static __global__ void ComplexPointwiseSub(const Complex* a, const Complex* b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexSub(a[i], b[i]);
    }
}

// Complex scalar scaling
static __global__ void ComplexScalarScale(const Complex* a, const float b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = ComplexScale(a[i], b);
    }
}

// Complex scalar adding
static __global__ void ComplexScalarAdd(const Complex* a, const float b, Complex* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i].x = a[i].x + b;
        c[i].y = a[i].y;
    }
}

// Complex scalar scaling
static __global__ void ComplexFill(Complex* a, const Complex b, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        a[i].x = b.x;
        a[i].y = b.y;
    }
}

// Complex scalar scaling
static __global__ void ComplexConvert(const float* a, Complex* b, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        b[i].x = a[i];
        b[i].y = 0;
    }
}

// Complex pointwise multiplication
static __global__ void RealPointwiseMul(const float* a, const float* b, float* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = a[i] * b[i];
    }
}

// Complex scalar multiplication
static __global__ void RealScalarMul(const float* a, const float b, float* c, int size) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        c[i] = a[i] * b;
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

    buildGaussian(idealResponse);

    cudaMalloc(&dSrc, realSize);
    cudaMalloc(&dKernel, realSize);
    cudaMalloc(&dGaussian, realSize);
    cudaMalloc(&dRelMap, realSize);
    cudaMalloc(&dSrcFFT, complexSize);
    cudaMalloc(&dKernelFFT, complexSize);
    cudaMalloc(&dGaussianFFT, complexSize);
    cudaMalloc(&dLagrangianFFT, complexSize);
    cudaMalloc(&dConv1, complexSize);
    cudaMalloc(&dConv2, complexSize);
    cudaMalloc(&dTemp1, complexSize);
    // cudaMalloc(&dTemp2, complexSize);
    
    // Apply FFT to idealResponse
    cudaMemcpy2D(dGaussian, bbox.width * sizeof(float), idealResponse.data, bbox.width * sizeof(float), bbox.width * sizeof(float), bbox.height, cudaMemcpyHostToDevice);
    cufftHandle plan1;
    cufftPlan2d(&plan1, bbox.height, bbox.width, CUFFT_R2C);
    cufftExecR2C(plan1, dGaussian, dSrcFFT);
    cufftDestroy(plan1);


    // Temporary use constant weights
    channelWeights[0] = 1.0f;
    channelWeights[1] = 1.0f;
    channelWeights[2] = 1.0f;
}


TrackerCSRT::~TrackerCSRT() {
    cudaFree(dSrc);
    cudaFree(dKernel);
    cudaFree(dGaussian);
    cudaFree(dRelMap);
    cudaFree(dSrcFFT);
    cudaFree(dKernelFFT);
    cudaFree(dGaussianFFT);
    cudaFree(dLagrangianFFT);
    cudaFree(dConv1);
    cudaFree(dConv2);
    cudaFree(dTemp1);
    // cudaFree(dTemp2);
}


bool TrackerCSRT::update(const cv::Mat& frame, cv::Rect& bbox) {
    updateChannels(frame, bbox);

    updateLocation(bbox);

    updateChannels(frame, bbox);
    
    updateFilter();

    return true;
}


void TrackerCSRT::updateChannels(const cv::Mat& frame, const cv::Rect& bbox) {
    std::vector<cv::Mat1f> temp(3);
    cv::split(cv::Mat(frame, bbox), temp.data());

    for (int channelId = 0; channelId < channels.size(); ++ channelId) {
        temp[channelId].convertTo(channels[channelId], CV_32F);
    }
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
    ComplexPointwiseMulConjScale<<<32, 256>>>(dSrcFFT, dKernelFFT, dSrcFFT, width * height, 1.0f / (width * height * width * height * width * height));

    // Apply IFFT to srcTemp
    cufftHandle plan2;
    cufftPlan2d(&plan2, height, width, CUFFT_C2R);
    cufftExecC2R(plan2, dSrcFFT, dSrc);
    cufftDestroy(plan2);

    // Copy result to host
    dst = cv::Mat1f::zeros(height, width);
    cudaMemcpy2D(dst.data, width * sizeof(float), dSrc, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
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

    double minVal;
    double maxVal;
    double weightsSum = 0;
    cv::Point minLoc;
    cv::Point maxLoc;

    for (int channelId = 0; channelId < channels.size(); ++channelId) {
        convolveCUDA(channels[channelId], filters[channelId], channelConvolution);

        // Add weighted channel convolution to resulting response
        resultingConvolution = resultingConvolution + channelConvolution * channelWeights[channelId];

        // Update weight by channel learning reliability
        cv::minMaxLoc(channelConvolution, &minVal, &maxVal, &minLoc, &maxLoc);
        std::cout << minVal << ' ' << maxVal << std::endl;
        channelWeights[channelId] = (1 - filterAdaptationRate) * channelWeights[channelId] + filterAdaptationRate * maxVal;
        weightsSum += channelWeights[channelId];
    }

    // Normalize weights
    for (int channelId = 0; channelId < channels.size(); ++channelId) {
        channelWeights[channelId] /= weightsSum;
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
    // cv::Mat1f maskedFilter;
    // cv::Mat1f constrainedFilter;
    // cv::Mat1f temp1;
    // cv::Mat1f temp2;

    cv::Mat relMap;
    estimateReliabilityMap(relMap);

    // int mu = 5;
    // const int beta = 3;
    // const float lambda = 0.01;
    // const int D = relMap.rows * relMap.cols;
    // const float denomIncrement = lambda / (2 * D);

    for (int channelId = 0; channelId < channels.size(); ++channelId) {
        // cv::Mat1f filter = filters[channelId].clone();
        // cv::Mat1f trainROI = channels[channelId];
        // cv::Mat1f L = cv::Mat1f::zeros(filter.rows, filter.cols);

        // int iter = 3;

        // while (iter--) {
        //     convolveCUDA(trainROI, idealResponse, temp1);
        //     convolveCUDA(trainROI, trainROI, temp2);

        //     cv::multiply(filter, relMap, maskedFilter);
        //     cv::divide(temp1 + (filterAdaptationRate * maskedFilter - L), temp2 + mu, constrainedFilter);
            
        //     cv::idft(L + mu * constrainedFilter, temp1, 0, filter.rows);
        //     cv::multiply(relMap, temp1, temp2);
        //     filter = temp2 / (denomIncrement + mu);

        //     L = L + filterAdaptationRate * (constrainedFilter - filter);

        //     mu *= beta;
        // }

        cv::Mat1f filter;
        estimateFilter(channelId, relMap, filter);

        // TEMP CONVERT FILTERS TO COMMON RANGE
        // cv::normalize(filters[channelId], filters[channelId], 0, 1, cv::NORM_MINMAX);
        // cv::normalize(filter, filter, 0, 255, cv::NORM_MINMAX);
        // cv::multiply(filter, relMap, filter);

        filters[channelId] = (1 - filterAdaptationRate) * filters[channelId] + filterAdaptationRate * filter;

        cv::Mat temp3;
        cv::normalize(filter, temp3, 0, 1, cv::NORM_MINMAX);
        cv::imshow("Filter", temp3);
        cv::waitKey(0);

        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        cv::minMaxLoc(filter, &minVal, &maxVal, &minLoc, &maxLoc);
        std::cout << minVal << ' ' << maxVal << std::endl;
        cv::minMaxLoc(filters[channelId], &minVal, &maxVal, &minLoc, &maxLoc);
        std::cout << minVal << ' ' << maxVal << std::endl;
        cv::minMaxLoc(channels[channelId], &minVal, &maxVal, &minLoc, &maxLoc);
        std::cout << minVal << ' ' << maxVal << std::endl;
    }

    // Show filter
    // cv::Mat temp3;
    // cv::normalize(filters[0], temp3, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("Filter", temp3);
}


void TrackerCSRT::estimateFilter(const int channelId, const cv::Mat1f& relMap, cv::Mat1f& newFilter) {
    int mu = 5;
    const int beta = 3;
    const float lambda = 0.01;
    const int D = relMap.rows * relMap.cols;
    const float denomIncrement = lambda / (2 * D);

    cv::Mat1f oldFilter = filters[channelId];
    cv::Mat1f trainROI = channels[channelId];

    size_t width = oldFilter.cols;
    size_t height = oldFilter.rows;

    float* dOldFilter = dSrc;
    float* dTrainROI = dKernel;
    Complex* dOldFilterFFT = dSrcFFT;
    Complex* dTrainROIFFT = dKernelFFT;

    // Copy src and kernel to device
    cudaMemcpy2D(dOldFilter, width * sizeof(float), oldFilter.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dTrainROI, width * sizeof(float), trainROI.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dRelMap, width * sizeof(float), relMap.data, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Initialize filter FFT with old filter
    ComplexConvert<<<32, 256>>>(dOldFilter, dOldFilterFFT, width * height);

    // Initialize Lagrangian FFT with zeros
    ComplexFill<<<32, 256>>>(dLagrangianFFT, {0, 0}, width * height);

    // Create FFT and IFFT plans
    cufftHandle plan1;
    cufftPlan2d(&plan1, height, width, CUFFT_R2C);
    cufftHandle plan2;
    cufftPlan2d(&plan2, height, width, CUFFT_C2R);

    // Apply FFT to srcTemp and kernel
    cufftExecR2C(plan1, dTrainROI, dTrainROIFFT);

    // Multiply elementwise srcTemp and kernel in Fourier domain
    ComplexPointwiseMulConjScale<<<32, 256>>>(dTrainROIFFT, dGaussianFFT, dConv1, width * height, 1.0f / (width * height * width * height * width * height));
    ComplexPointwiseMulConjScale<<<32, 256>>>(dTrainROIFFT, dTrainROIFFT, dConv2, width * height, 1.0f / (width * height * width * height * width * height));

    int iter = 1;

    while (iter--) {
        // Masked filter
        // ComplexPointwiseScale<<<32, 256>>>(dOldFilterFFT, dRelMap, dOldFilterFFT, width * height);
        RealPointwiseMul<<<32, 256>>>(dOldFilter, dRelMap, dOldFilter, width * height);
        cufftExecR2C(plan1, dOldFilter, dOldFilterFFT);

        // Calculate constrained filter into dOldFilterFFT
        ComplexScalarScale<<<32, 256>>>(dOldFilterFFT, filterAdaptationRate, dOldFilterFFT, width * height);
        ComplexPointwiseSub<<<32, 256>>>(dOldFilterFFT, dLagrangianFFT, dOldFilterFFT, width * height);
        ComplexPointwiseAdd<<<32, 256>>>(dConv1, dOldFilterFFT, dOldFilterFFT, width * height);
        ComplexScalarAdd<<<32, 256>>>(dConv2, mu, dTemp1, width * height);
        ComplexPointwiseDiv<<<32, 256>>>(dOldFilterFFT, dTemp1, dOldFilterFFT, width * height);

        // Calculate new filter into dOldFilter
        ComplexScalarScale<<<32, 256>>>(dOldFilterFFT, mu, dTemp1, width * height);
        ComplexPointwiseAdd<<<32, 256>>>(dLagrangianFFT, dTemp1, dTemp1, width * height);

        // Apply IFFT to srcTemp
        cufftExecC2R(plan2, dTemp1, dOldFilter);

        RealPointwiseMul<<<32, 256>>>(dRelMap, dOldFilter, dOldFilter, width * height);
        RealScalarMul<<<32, 256>>>(dOldFilter, 1 / (denomIncrement + mu), dOldFilter, width * height);
/*
        // Update Lagrangian
        cufftExecR2C(plan1, dOldFilter, dTemp1);
        ComplexPointwiseSub<<<32, 256>>>(dOldFilterFFT, dTemp1, dTemp1, width * height);
        ComplexScalarScale<<<32, 256>>>(dTemp1, filterAdaptationRate, dTemp1, width * height);
        ComplexPointwiseAdd<<<32, 256>>>(dLagrangianFFT, dTemp1, dLagrangianFFT, width * height);

        mu *= beta; */
    }

    cufftDestroy(plan1);
    cufftDestroy(plan2);

    // Copy result to host
    newFilter = cv::Mat1f::zeros(height, width);
    cudaMemcpy2D(newFilter.data, width * sizeof(float), dOldFilter, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
}
