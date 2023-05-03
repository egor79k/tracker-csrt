# tracker-csrt
Tracking using Discriminative Correlation Filter with Channel and Spatial Reliability

### Demo
Build:
```
cd tracker-csrt
mkdir build
cd build
cmake ..
make
```
Run:
```
./test_csrt <input video> [output video]
```

### Usage example:
Add in project CMakeLists:
```
include_directories(src)
...
target_link_libraries(<project> csrt)
```
Tracking in the video stream example:
```cpp
#include <opencv2/opencv.hpp>
#include <tracker_csrt.hpp>

...

cv::VideoCapture video("input_video.mp4");
cv::Mat frame;
video.read(frame)
cv::Rect bbox = cv::selectROI(frame, false);

TrackerCSRT tracker(frame, bbox);

while(video.read(frame)) {
    if (tracker.update(frame, bbox)) {
         // Handle new bbox
    }
}
```

### Links
- [Original article](https://arxiv.org/pdf/1611.08461v2.pdf)
