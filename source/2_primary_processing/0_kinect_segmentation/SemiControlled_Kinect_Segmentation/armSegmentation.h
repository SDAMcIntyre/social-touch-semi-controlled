#ifndef ARMSEGMENTATION_H
#define ARMSEGMENTATION_H

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <vector>

// Point Cloud Library
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/point_types_conversion.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>


static int findMinMaxXYZ(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

class ArmSegmentation {
public:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_capture;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_arm;

    bool show;

    // Constructor that takes the MKV file location as input
    ArmSegmentation();

    // Destructor to release resources
    ~ArmSegmentation();

    void showProcessingSteps(bool show_);

    // initialisation setters, to be used before preprocessing
    void set_pointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc);
    void use_boxFilter(bool use_boxFilter_, const float box_corners_[2][3] = 0);
    void use_downSampling(bool use_downSampling_, int downSampling_);
    void use_colorSkinFilter(bool use_colorSkinFilter_, const float* hsv_lowerBound = nullptr, const float* hsv_upperBound = nullptr);

    void set_segmentationThreshold(double threshold);

    // 
    bool isNonSkinPixel(pcl::PointXYZHSV point);

    // 
    int preprocess();

    // 
    int extract_pointCloudArm();

private:
    // PointCloud working material
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_modified;

    // private functions and properties
    bool useBoxFilter;
    float boxFilter_corners[2][3];

    bool useDownSampling;
    int downSampling_value;
    
    bool useColorSkinFilter;
    float colorSkin_hsv_lb[3];
    float colorSkin_hsv_HB[3];

    double seg_threshold;
};

#endif // ARMSEGMENTATION_H

