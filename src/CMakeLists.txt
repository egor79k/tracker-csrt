find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

# add_executable(csrt_opencv csrt_opencv.cpp)
# target_link_libraries(csrt_opencv ${OpenCV_LIBS})

add_library(csrt tracker_csrt.cu)
target_link_libraries(csrt cufft ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_csrt main.cpp)
target_link_libraries(test_csrt csrt)
