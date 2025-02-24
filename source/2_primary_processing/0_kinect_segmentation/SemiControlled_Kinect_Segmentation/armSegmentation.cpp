#include "armSegmentation.h"


// Constructor
ArmSegmentation::ArmSegmentation() {
    this->pc_capture = nullptr;
    this->pc_arm = nullptr;
    this->pc_modified = nullptr;

    this->show = false;

    this->useBoxFilter = false;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            this->boxFilter_corners[i][j] = 0;
        }
    }

    this->useDownSampling = false;
    this->downSampling_value = 0;

    this->useColorSkinFilter = false;
    for (int j = 0; j < 3; ++j) {
        this->colorSkin_hsv_lb[j] = 0;
        this->colorSkin_hsv_HB[j] = 0;
    }

    this->seg_threshold = -1;
}

// Destructor
ArmSegmentation::~ArmSegmentation() {

}

void ArmSegmentation::showProcessingSteps(bool show_) {
    this->show = show_;
}

void ArmSegmentation::set_pointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc) {
    this->pc_capture = pc;
}

void ArmSegmentation::use_boxFilter(bool use_boxFilter_, const float box_corners_[2][3]) {
    this->useBoxFilter = use_boxFilter_;
    if (this->useBoxFilter) {
        // Copy the contents of box_corners_ to this->box_corners
        for (int corner = 0; corner < 2; ++corner) {
            for (int dim = 0; dim < 3; ++dim) {
                this->boxFilter_corners[corner][dim] = box_corners_[corner][dim];
            }
        }
    }
}

void ArmSegmentation::use_downSampling(bool use_downSampling_, int downSampling_) {
    this->useDownSampling = use_downSampling_;
    if (this->useDownSampling) {
        this->downSampling_value = downSampling_;
    }
}


void ArmSegmentation::use_colorSkinFilter(bool use_colorSkinFilter_, const float* hsv_lowerBound, const float* hsv_upperBound) {
    this->useColorSkinFilter = use_colorSkinFilter_;
    if (this->useColorSkinFilter) {
        // Default values if pointers are null
        const float default_lower[3] = { 0.0f, 1.0f, 0.95f };
        const float default_upper[3] = { 40.0f, 1.0f, 1.0f };

        // Copy the contents of hsv_lowerBound or use default values
        if (hsv_lowerBound) {
            for (int j = 0; j < 3; ++j) {
                this->colorSkin_hsv_lb[j] = hsv_lowerBound[j];
            }
        }
        else {
            for (int j = 0; j < 3; ++j) {
                this->colorSkin_hsv_lb[j] = default_lower[j];
            }
        }

        // Copy the contents of hsv_upperBound or use default values
        if (hsv_upperBound) {
            for (int j = 0; j < 3; ++j) {
                this->colorSkin_hsv_HB[j] = hsv_upperBound[j];
            }
        }
        else {
            for (int j = 0; j < 3; ++j) {
                this->colorSkin_hsv_HB[j] = default_upper[j];
            }
        }
    }
}

void ArmSegmentation::set_segmentationThreshold(double threshold) {
    this->seg_threshold = threshold;
}

int ArmSegmentation::preprocess() {
    int error_code = 0;

    this->pc_modified.reset();
    this->pc_modified = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*(this->pc_capture), *(this->pc_modified));
    if (this->show) {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Input Point Cloud"));
        viewer->addPointCloud(this->pc_modified);
        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
    }

    // Down Sampling
    if (useDownSampling) {
        const float voxel_grid_size = this->downSampling_value;
        pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
        vox_grid.setInputCloud(this->pc_modified);
        vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        vox_grid.filter(*(this->pc_modified));
        std::cout << "PointCloud, after downsample, has: " << this->pc_modified->points.size() << " data points." << std::endl; //*
        
        if (this->show) {
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Downsampled Point Cloud"));
            viewer->addPointCloud(this->pc_modified);

            // Set the camera position to unzoom by 500
            viewer->setCameraPosition(0, 0, 500, 0, 0, 1);

            while (!viewer->wasStopped()) {
                viewer->spinOnce();
            }
        }

    }

    // Box Crop Filter
    if (useBoxFilter) {
        pcl::CropBox<pcl::PointXYZRGB> boxFilter;

        std::cout << "Applying 3D box filtering... PointCloud has " << this->pc_modified->points.size() << " data points." << std::endl;
        for (int i = 0; i < 2; ++i) {
            std::cout << "Point " << i + 1 << ": "
                << "X = " << this->boxFilter_corners[i][0] << ", "
                << "Y = " << this->boxFilter_corners[i][1] << ", "
                << "Z = " << this->boxFilter_corners[i][2] << std::endl;
        }

        boxFilter.setMin(Eigen::Vector4f(this->boxFilter_corners[0][0], this->boxFilter_corners[0][1], 0, 1.0));
        boxFilter.setMax(Eigen::Vector4f(this->boxFilter_corners[1][0], this->boxFilter_corners[1][1], 5000, 1.0));
        boxFilter.setInputCloud(this->pc_modified);
        boxFilter.filter(*(this->pc_modified));
        std::cout << "PointCloud, after filtering with a box around the arm, has: " << this->pc_modified->points.size() << " data points." << std::endl;
        
        // Visualization
        if (this->show) {
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("After applying the box filter"));
            viewer->addPointCloud(this->pc_modified);
            while (!viewer->wasStopped()) {
                viewer->spinOnce();
            }
        }
    }

    /// extract the arm by color in HSV color space in a 1D vector
    if (useColorSkinFilter) {
        pcl::PointCloud<pcl::PointXYZHSV>::Ptr Cloud_HSV(new pcl::PointCloud<pcl::PointXYZHSV>);
        pcl::PointCloudXYZRGBtoXYZHSV(*(this->pc_modified), *Cloud_HSV);

        std::cout << "Applying color skin filtering... PointCloud has " << this->pc_modified->points.size() << " data points." << std::endl;

        float hsv_lb_local[3];
        for (int i = 0; i < 3; ++i) {
            hsv_lb_local[i] = this->colorSkin_hsv_lb[i];
        }
        float hsv_HB_local[3];
        for (int i = 0; i < 3; ++i) {
            hsv_HB_local[i] = this->colorSkin_hsv_HB[i];
        }
        Cloud_HSV->points.erase(std::remove_if(Cloud_HSV->points.begin(), Cloud_HSV->points.end(),
                                               [hsv_lb_local, hsv_HB_local](pcl::PointXYZHSV point) {
                                                    return !((hsv_lb_local[0] < point.h && point.h < hsv_HB_local[0]) || hsv_lb_local[2] < point.v); }), 
                                Cloud_HSV->points.end());
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Cloud_RGB(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloudXYZHSVtoXYZRGB(*Cloud_HSV, *Cloud_RGB);
        *(this->pc_modified) = *Cloud_RGB;
        std::cout << "Erased non-skin pixel PointCloud has now " << pc_modified->points.size() << " data points." << std::endl;

        // Visualization
        if (this->show) {
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("After filtering by skin color"));
            viewer->addPointCloud(this->pc_modified);
            while (!viewer->wasStopped()) {
                viewer->spinOnce();
            }
        }
    }

    return error_code;
}


int ArmSegmentation::extract_pointCloudArm() {
    int error_code = 0;

    this->pc_arm.reset();
    this->pc_arm = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

    //////---------Growing Region Segmentation----------
    // Normal
    pcl::search::Search<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);

    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(this->pc_modified);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    // Growing Segmentation
    pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
    reg.setMinClusterSize(500);
    reg.setMaxClusterSize(1000000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(this->pc_modified);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(this->seg_threshold / 180.0 * M_PI);  // 4 for p1,p2; 4.5 or 3.5 for p3
    reg.setCurvatureThreshold(1.0);
    std::vector <pcl::PointIndices> clusters;
    reg.extract(clusters);
    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    //std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
    std::vector<int> ClusterSize;
    for (int i = 0; i < clusters.size(); i++)
        ClusterSize.push_back(clusters[i].indices.size());
    std::vector<int>::iterator cluster_biggest_it = std::max_element(std::begin(ClusterSize), std::end(ClusterSize));
    int cluster_biggest_idx = std::distance(std::begin(ClusterSize), cluster_biggest_it);
    std::cout << "Biggest cluster has " << *cluster_biggest_it << " points, at position " << cluster_biggest_idx << std::endl;
    // Visualization
    if (show) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
        pcl::visualization::CloudViewer viewer("Cluster viewer");
        viewer.showCloud(colored_cloud);
        while (!viewer.wasStopped()) {}
    }

    // Pick the arm cluster
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(this->pc_modified);
    pcl::PointIndicesPtr pi_ptr(new pcl::PointIndices);
    pi_ptr->indices = clusters[cluster_biggest_idx].indices;    //2 for p1   1 for p2   0 for p3          
    extract.setIndices(pi_ptr);
    extract.filter(*(this->pc_arm));
    std::cout << "PointCloud of arm has: " << this->pc_arm->points.size() << " data points." << std::endl;
    // Visualization
    if (show) {
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("arm viewer"));
        viewer->addPointCloud(this->pc_arm);
        while (!viewer->wasStopped()) {
            viewer->spinOnce();
        }
    }

    // Normal vectors for arm cluster
    pcl::PointCloud <pcl::Normal>::Ptr arm_normals(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> arm_normal_estimator;
    arm_normal_estimator.setSearchMethod(tree);
    arm_normal_estimator.setInputCloud(this->pc_arm);
    arm_normal_estimator.setKSearch(50);
    arm_normal_estimator.compute(*arm_normals);

    return error_code;
}


static int findMinMaxXYZ(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    // Initialize min and max values
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();

    // Iterate over all points in the cloud
    for (const auto& point : cloud->points) {
        if (point.x < min_x) min_x = point.x;
        if (point.x > max_x) max_x = point.x;

        if (point.y < min_y) min_y = point.y;
        if (point.y > max_y) max_y = point.y;

        if (point.z < min_z) min_z = point.z;
        if (point.z > max_z) max_z = point.z;
    }

    // Print the results
    std::cout << "Min X: " << min_x << " Max X: " << max_x << std::endl;
    std::cout << "Min Y: " << min_y << " Max Y: " << max_y << std::endl;
    std::cout << "Min Z: " << min_z << " Max Z: " << max_z << std::endl;

    // Return the results as a tuple (min_x, max_x, min_y, max_y, min_z, max_z)
    return 0;
}