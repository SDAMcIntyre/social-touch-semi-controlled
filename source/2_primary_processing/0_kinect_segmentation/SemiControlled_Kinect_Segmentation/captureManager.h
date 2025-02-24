#ifndef CAPTUREMANAGER_H
#define CAPTUREMANAGER_H

#include "vtkAutoInit.h" 
#define vtkRenderingCore_AUTOINIT 2(vtkRenderingOpenGL2, vtkInteractionStyle)

#include <string>
#include <vector>

// Include the necessary headers for working with Kinect files and channels
#include <k4a/k4a.hpp>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4arecord/record.h>
#include "turbojpeg.h"

// to generate the pointCloud
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// to plot images
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

class CaptureManager {
public:
    k4a_playback_t playback;
    k4a_capture_t capture;
    int frame_id;

    k4a_image_t color;
    k4a_image_t depth;
    k4a_image_t depth_colorpov;
    k4a_image_t depth_xyz_colorpov;

    k4a_result_t result;
    k4a_stream_result_t stream_result;

    // Constructor that takes the MKV file location as input
    CaptureManager(const std::string& mkvFilePath, int _frame_id = -1, k4a_playback_t _playback = nullptr);

    // Destructor to release resources
    ~CaptureManager();

    // Method to load the MKV file 
    int initialize();

    // Method to load color channel
    int loadImage();
    // Method to load depth channel
    int loadDepth();

    // Method to create depth channel with the resolution of the color camera
    int getDepth_colorCameraPov();
    // Method to create depth image with 3 mm values (XYZ) for each pixel of the color camera (1920*1080).
    int getDepth_xyz_colorCameraPov();

    //
    int generate_pointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud);

    // get XYZ mm location from XY pixel coordinate
    int convert_XYcoordinate_to_XYZmm(int xy[2], float xyz[3]);

    // get the XY pixel that has a depth information
    k4a_float3_t findNonZeroZ(const k4a_float2_t& point2d);

    // Methods to plot images
    int plotColor(bool blocking = false);
    int plotDepth(bool blocking = false);
    int plotDepth_colorpov(bool blocking = false);
    void plotImage(void* buffer, int width, int height, int type, std::string title, bool blocking = false);
    void getDepth_as_rgb(std::vector<uint8_t> &rgbImage);

private:
    std::string mkvFilePath;
    // Method to clean up and close resources
    void cleanup();
    void imageGrayscaleToRGB(const uint16_t* gray, uint8_t* rgb, int width, int height);
    void huetoRGB(float& fH, float& fR, float& fG, float& fB);
};

#endif // CAPTUREMANAGER_H

