// Standard library headers
#include <iostream>          // For std::cerr, std::cout
#include <fstream>           // For std::ifstream
#include <string>            // For std::string

// imported libraries
#include <nlohmann/json.hpp>

// custom libraries
#include "captureManager.h"
#include "armSegmentation.h"

#include <filesystem>
#include <boost/locale.hpp>
#include <commdlg.h>
#include <vector>
#include <windows.h>
#include <shlobj.h> // Include this header for BROWSEINFO, LPITEMIDLIST, SHBrowseForFolder, SHGetPathFromIDList


// for convenience
using json = nlohmann::json;
using namespace std::chrono_literals;
using namespace cv;

struct RegionOfInterest {
	int top_left_x;
	int top_left_y;
	int bottom_right_x;
	int bottom_right_y;
};

struct VideoData {
	std::string video_path;
	int reference_frame_idx;
	RegionOfInterest roi;
	int frame_width;
	int frame_height;
	double fps;
};

std::vector<std::filesystem::path> extractMetadataFilenames(std::filesystem::path directory_target = "");
VideoData extractVideoData(const std::string& file_path);
int processFile(const std::string& input_file_path, bool show, bool show_result);


int main(int argc, char* argv[]) {
	int error_code = 0;
	bool show_result = false;
	bool show = false;
	bool select_directory = false;

	std::filesystem::path file_path;
	std::vector<std::filesystem::path> filenames;
	// Define your specific path here
	// std::filesystem::path directory_target = ""
	std::filesystem::path directory_target = "F:\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\1_primary\\kinect\\2_arm_roi";

	// Parse command-line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--show") {
			show = true;
		}
		else if (arg == "--show_result") {
			show_result = true;
		}
	}

	// Open file dialog to select files
	filenames = extractMetadataFilenames(directory_target);

	for (const auto& filepath : filenames) {
		int error_code = processFile(filepath.string(), show, show_result);
		if (error_code) {
			std::cerr << "Error processing file: " << filepath << " with error code: " << error_code << std::endl;
		}
	}

Exit:
	return error_code;
}





int processFile(const std::string& input_file_path, bool show, bool show_result) {
	int error_code = 0;
	std::filesystem::path file_path;
	std::filesystem::path directory;

	// Generate output file path
	std::string output_file_path = input_file_path;
	size_t pos = output_file_path.find("2_arm_roi");
	if (pos != std::string::npos) {
		output_file_path.replace(pos, std::string("2_arm_roi").length(), "2_arm_pointCloud");
	}
	pos = output_file_path.find("_roi_metadata.txt");
	if (pos != std::string::npos) {
		output_file_path.replace(pos, std::string("_roi_metadata.txt").length(), ".ply");
	}
	std::replace(output_file_path.begin(), output_file_path.end(), 'Ã', 'ö');
	std::replace(output_file_path.begin(), output_file_path.end(), '÷', 'ö');
	output_file_path.erase(std::remove(output_file_path.begin(), output_file_path.end(), '¶'), output_file_path.end());
	std::cout << "output_file_path: " << output_file_path << "\n";

	// Extract video data from JSON
	VideoData data = extractVideoData(input_file_path);

	// Print extracted data
	std::cout << "Video Path: " << data.video_path << "\n";
	std::cout << "Reference Frame Index: " << data.reference_frame_idx << "\n";
	std::cout << "ROI Top Left: (" << data.roi.top_left_x << ", " << data.roi.top_left_y << ")\n";
	std::cout << "ROI Bottom Right: (" << data.roi.bottom_right_x << ", " << data.roi.bottom_right_y << ")\n";
	std::cout << "Frame Width: " << data.frame_width << "\n";
	std::cout << "Frame Height: " << data.frame_height << "\n";
	std::cout << "FPS: " << data.fps << "\n";

	// Work variables
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_capture(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_arm(new pcl::PointCloud<pcl::PointXYZRGB>);
	int xy[2]; // Opposed points of the rectangle targeting the area of contact
	// Parameters for Point Cloud segmentation
	float cuboid_xyz[2][3];
	const float lowerBound[3] = { 0.0f, 1.0f, 0.95f };  // Define arrays for color skin filter bounds
	const float upperBound[3] = { 40.0f, 1.0f, 1.0f };  // Define arrays for color skin filter bounds
	double seg_thre = 9.0;  // Segmentation parameters

	CaptureManager cm(data.video_path, data.reference_frame_idx);
	ArmSegmentation segmenter;

	error_code = cm.initialize();
	if (error_code) goto Exit;
	error_code = cm.loadImage();
	if (error_code) goto Exit;
	error_code = cm.loadDepth();
	if (error_code) goto Exit;
	error_code = cm.getDepth_colorCameraPov();
	if (error_code) goto Exit;
	error_code = cm.getDepth_xyz_colorCameraPov();
	if (error_code) goto Exit;
	// Get the XYZRGB list of valid points in the capture.
	error_code = cm.generate_pointCloud(pc_capture);
	if (error_code) goto Exit;

	// Convert xy pixels (2D color image) into xyz mm (3D Kinect space) for pcl
	xy[0] = data.roi.top_left_x;
	xy[1] = data.roi.top_left_y;
	cm.convert_XYcoordinate_to_XYZmm(xy, cuboid_xyz[0]);
	xy[0] = data.roi.bottom_right_x;
	xy[1] = data.roi.bottom_right_y;
	cm.convert_XYcoordinate_to_XYZmm(xy, cuboid_xyz[1]);

	segmenter.showProcessingSteps(show);
	segmenter.set_pointCloud(pc_capture);
	segmenter.use_downSampling(false, 5);
	segmenter.use_boxFilter(true, cuboid_xyz);
	segmenter.use_colorSkinFilter(true, lowerBound, upperBound);
	error_code = segmenter.preprocess();
	if (error_code) goto Exit;

	segmenter.set_segmentationThreshold(seg_thre);
	error_code = segmenter.extract_pointCloudArm();
	if (error_code) goto Exit;

	// Display with OpenCV
	if (show_result) {
		// Show final point cloud
		pcl::visualization::CloudViewer arm_viewer("arm viewer");
		arm_viewer.showCloud(segmenter.pc_arm);

		// Display 2D images
		std::vector<uint8_t> rgbImage;
		uint8_t* buffer;
		int w, h;
		// Convert the box_xy to cv::Point for cv::rectangle
		cv::Point pt1(static_cast<int>(data.roi.top_left_x), static_cast<int>(data.roi.top_left_y));
		cv::Point pt2(static_cast<int>(data.roi.bottom_right_x), static_cast<int>(data.roi.bottom_right_y));
		// Depth and box
		w = k4a_image_get_width_pixels(cm.depth_colorpov);
		h = k4a_image_get_height_pixels(cm.depth_colorpov);
		cm.getDepth_as_rgb(rgbImage);
		cv::Mat depthMat(h, w, CV_8UC3, rgbImage.data(), cv::Mat::AUTO_STEP);
		// Add the rectangle to bgra_mat
		cv::rectangle(depthMat, pt1, pt2, cv::Scalar(0, 0, 255), 2); // Red rectangle with thickness 2
		// Display the image with the rectangle
		cv::imshow("kinect depth frame", depthMat);

		// Color and box
		buffer = k4a_image_get_buffer(cm.color);
		w = k4a_image_get_width_pixels(cm.color);
		h = k4a_image_get_height_pixels(cm.color);
		cv::Mat colorMat(h, w, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);
		// Add the rectangle to bgra_mat
		cv::rectangle(colorMat, pt1, pt2, cv::Scalar(0, 0, 255), 2); // Red rectangle with thickness 2
		// Display the image with the rectangle
		cv::imshow("kinect color frame", colorMat);

		cv::waitKey(0); // Wait for a key press to close the window
	}

	// Save result
	file_path = output_file_path;
	directory = file_path.parent_path();
	if (!std::filesystem::exists(directory)) {
		std::filesystem::create_directories(directory);
	}
	std::cout << "output directory: " << directory << std::endl;
	pcl::io::savePLYFile<pcl::PointXYZRGB>(output_file_path, *(segmenter.pc_arm));

Exit:
	return error_code;
}


VideoData extractVideoData(const std::string& file_path) {
	VideoData data;

	std::string filep(file_path);
	// Open the JSON file
	std::ifstream file(file_path);

	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + file_path);
	}

	std::stringstream stream;
	stream << file.rdbuf();
	std::string json_string = stream.str();

	//cout << "File contents:" << endl << json_string << endl << endl;

	// Parse the JSON content
	json j = json::parse(json_string);  // Corrected line

	// Extract data
	data.video_path = j.at("video_path").get<std::string>();
	// hotfix for difficulties to read the ö symbol:  Replace 'Ã' with 'ö' and remove all occurrences of '¶'
	std::replace(data.video_path.begin(), data.video_path.end(), 'Ã', 'ö');
	std::replace(data.video_path.begin(), data.video_path.end(), '÷', 'ö');
	data.video_path.erase(std::remove(data.video_path.begin(), data.video_path.end(), '¶'), data.video_path.end());

	// Extract 
	data.reference_frame_idx = j.at("reference_frame_idx").get<int>();

	// Extract the region of interest
	data.roi.top_left_x = j.at("region_of_interest").at("top_left_corner").at("x").get<int>();
	data.roi.top_left_y = j.at("region_of_interest").at("top_left_corner").at("y").get<int>();
	data.roi.bottom_right_x = j.at("region_of_interest").at("bottom_right_corner").at("x").get<int>();
	data.roi.bottom_right_y = j.at("region_of_interest").at("bottom_right_corner").at("y").get<int>();

	data.frame_width = j.at("frame_width").get<int>();
	data.frame_height = j.at("frame_height").get<int>();
	data.fps = j.at("fps").get<double>();


	return data;
}


// Function to open file dialog
std::vector<std::filesystem::path> extractMetadataFilenames(std::filesystem::path directory_target){
	OPENFILENAME ofn;
	char szFile[260] = { 0 };
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "Text Files\0*.TXT\0All Files\0*.*\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	std::string input_path;
	std::filesystem::path directory;
	std::vector<std::filesystem::path> files;
	boolean selectDirectory = false;

	// Prompt user to choose between file and directory
	int msgboxID = MessageBox(
		NULL,
		"Do you want to select a directory (YES, otherwise a file is expected) ?",
		"Select Mode",
		MB_ICONQUESTION | MB_YESNO
	);
	if (msgboxID == IDYES) {
		selectDirectory = true;
	}

	if (selectDirectory) {
		// Initialize the BROWSEINFO structure
		BROWSEINFO bi = { 0 };
		bi.lpszTitle = "Select Directory";
		bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
		if (!directory_target.empty()) {
			// Convert the path to a PIDL
			LPITEMIDLIST pidlRoot = nullptr;
			HRESULT hr = SHParseDisplayName(directory_target.c_str(), nullptr, &pidlRoot, 0, nullptr);
			if (FAILED(hr)) {
				std::cerr << "Failed to parse directory path." << std::endl;
				return files;
			}
			bi.pidlRoot = pidlRoot;
		}

		// Display the directory selection dialog
		LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
		if (pidl != 0) {
			// Get the name of the folder
			SHGetPathFromIDList(pidl, szFile);
			directory = std::filesystem::path(szFile);
			for (const auto& entry : std::filesystem::directory_iterator(directory)) {
				if (entry.path().extension() == ".txt" && entry.path().filename().string().find("_roi_metadata") != std::string::npos) {
					files.push_back(entry.path());
				}
			}
		}
	}
	else {
		if (GetOpenFileName(&ofn) == TRUE) {
			input_path = std::string(ofn.lpstrFile);
			files.push_back(input_path);
		}
	}

	if (files.empty()) {
		std::cerr << "No _roi_metadata.txt file selected or found." << std::endl;
		return files;
	}

	// Process files
	for (const auto& file : files) {
		std::cout << "Processing file: " << file << std::endl;
		// Add your file processing code here
	}

	return files;
}

