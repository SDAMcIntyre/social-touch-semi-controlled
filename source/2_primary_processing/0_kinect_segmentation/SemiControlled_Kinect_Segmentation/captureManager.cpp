#include "CaptureManager.h"
#include <iostream>

// Constructor
CaptureManager::CaptureManager(const std::string & mkvFilePath, int _frame_id, k4a_playback_t _playback):
    mkvFilePath(mkvFilePath), frame_id(_frame_id), playback(_playback) {
    this->capture = nullptr;

    this->color = NULL;
    this->depth = NULL;
	this->depth_colorpov = NULL;
	this->depth_xyz_colorpov = NULL;
	
    this->result = K4A_RESULT_FAILED;
    this->stream_result = K4A_STREAM_RESULT_FAILED;
}

// Destructor
CaptureManager::~CaptureManager() {
    cleanup();
}

// Initialize the MKV file and load basic information
int CaptureManager::initialize() {
	int error_code = 0;
	int curr_frame_id = -1;

	if (this->playback == nullptr) {
		this->result = k4a_playback_open(&this->mkvFilePath[0], &this->playback);
		if (this->result != K4A_RESULT_SUCCEEDED || this->playback == NULL) {
			printf("Failed to open recording %s\n", &this->mkvFilePath[0]);
			error_code = 1;
			goto Exit;
		}
	}
	if (this->color != NULL) {
		k4a_image_release(this->color);
		this->color = NULL;  
	}

	if (this->depth != NULL) {
		k4a_image_release(this->depth);
		this->depth = NULL;  
	}

	// If frame id is requested, set up the playback function just before the specific frame 
	if (this->frame_id != -1) {
		while (curr_frame_id < frame_id-1) {
			this->stream_result = k4a_playback_get_next_capture(this->playback, &this->capture);
			curr_frame_id++;
		}
	}

	// find a frame that has both depth and color data
	while (this->stream_result == K4A_STREAM_RESULT_FAILED || this->color == NULL || this->depth == NULL) {
		// for some reason, log file generation triggers an exception, so try catch is necessary
		// https://learn.microsoft.com/en-us/azure/kinect-dk/troubleshooting
		// https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1987
		try {
			curr_frame_id++;
			// Get current frame (capture)
			std::cout << "reading... frame: " << this->frame_id << std::endl;
			this->stream_result = k4a_playback_get_next_capture(this->playback, &this->capture);
			if (stream_result == K4A_STREAM_RESULT_SUCCEEDED) {
				// to check if image handles exists
				this->color = k4a_capture_get_color_image(this->capture);
				this->depth = k4a_capture_get_depth_image(this->capture);
			}
		} 
		catch (const std::exception& e) {
			this->stream_result = K4A_STREAM_RESULT_FAILED;
		}
	}

	if (this->stream_result != K4A_STREAM_RESULT_SUCCEEDED || this->capture == nullptr) {
		printf("Failed to fetch frame\n");
		error_code = 1;
		goto Exit;
	}
	if (this->color == NULL) {
		printf("Failed to get color image from capture\n");
		error_code = 1;
		goto Exit;
	}
	else {
		k4a_image_release(this->color);
		this->color = NULL;  
	}

	if (this->depth == NULL) {
		printf("Failed to get depth image from capture\n");
		error_code = 1;
		goto Exit;
	}
	else {
		k4a_image_release(this->depth);
		this->depth = NULL; 
	}

	// save the frame id for later
	this->frame_id = curr_frame_id;

Exit:
    return error_code;
}

int CaptureManager::loadImage() {
	int error_code = 0;
	k4a_image_t color_tmp = NULL;  
	k4a_image_format_t format;

	int width, height;

	// make sure capture is available
	if (this->capture == nullptr) {
		printf("capture not found: initialise() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure that the output buffer is free
	if (this->color != NULL) {
		k4a_image_release(this->color);
		this->color = NULL;  // Reset to nullptr after release
	}

	// sanity check on the color image
	color_tmp = k4a_capture_get_color_image(this->capture);
	if (color_tmp == NULL) {
		printf("There is no color image in this capture.\n");
		error_code = 1;
		goto Exit;
	}
	width = k4a_image_get_width_pixels(color_tmp);
	height = k4a_image_get_height_pixels(color_tmp);
	if (width == 0 || height == 0) {
		printf("At least one dimension is equal to zero.\n");
		error_code = 2;
		goto Exit;
	}

	format = k4a_image_get_format(color_tmp);
	// if it is already in a non compressed format, do nothing
	if (format == K4A_IMAGE_FORMAT_COLOR_BGRA32) {
		this->color = color_tmp;
		goto Exit;
	}
	// if it is in the mpjeg compressed format, uncompress it using jpegturbo to bgra
	else if (format == K4A_IMAGE_FORMAT_COLOR_MJPG) {
		// create space in memory to receive the uncompressed image
		this->result = k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, width, height, 4 * width * (int)sizeof(uint8_t), &this->color);
		if (K4A_RESULT_SUCCEEDED != this->result) {
			printf("Failed to create image buffer\n");
			k4a_image_release(color_tmp);
			error_code = 3;
			goto Exit;
		}
		// Decompress the MPKEH into the color's buffer
		tjhandle tjHandle; // use of turbojpeg
		int tj_error;
		tjHandle = tjInitDecompress();
		tj_error = tjDecompress2(tjHandle, k4a_image_get_buffer(color_tmp), static_cast<unsigned long>(k4a_image_get_size(color_tmp)),
			k4a_image_get_buffer(this->color),
			width, 0, height, TJPF_BGRA, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE);
		// release the buffer of the compressed image
		k4a_image_release(color_tmp);
		if (tj_error) {
			printf("Failed to decompress color frame\n");
			if (tjDestroy(tjHandle)) {
				printf("Failed to destroy turboJPEG handle\n");
			}
			error_code = 3;
			goto Exit;
		}
		if (tjDestroy(tjHandle)) {
			printf("Failed to destroy turboJPEG handle\n");
		}
	}

Exit:
	return error_code;
}

int CaptureManager::loadDepth() {
	int error_code = 0;

	// make sure capture is available
	if (this->capture == nullptr) {
		printf("capture not found: initialise() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure that the output buffer is free
	if (this->depth != NULL) {
		k4a_image_release(this->depth);
		this->depth = NULL;  // Reset to NULL after release
	}

	this->depth = k4a_capture_get_depth_image(this->capture);
	if (this->depth == NULL) {
		printf("There is no depth image in this capture.\n");
		error_code = 1;
		goto Exit;
	}

Exit:
	return error_code;
}

// Method to create depth channel with the resolution of the color camera
int CaptureManager::getDepth_colorCameraPov() {
	int error_code = 0;
	k4a_calibration_t calibration;
	k4a_transformation_t transformation_handle = nullptr;
	int width, height;

	// make sure playback is available
	if (this->playback == nullptr) {
		printf("playback not found: initialise() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure color is available
	if (this->color == NULL) {
		printf("color image not found: loadColor() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure depth is available
	if (this->depth == NULL) {
		printf("depth image not found: loadDepth() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure that the output buffer is free
	if (this->depth_colorpov != NULL) {
		k4a_image_release(this->depth_colorpov);
		this->depth_colorpov = NULL;  // Reset to nullptr after release
	}

	// get transformation
	this->result = k4a_playback_get_calibration(this->playback, &calibration);
	if (K4A_RESULT_SUCCEEDED != this->result) {
		printf("Failed to get calibration\n");
		error_code = 1;
		goto Exit;
	}
	transformation_handle = k4a_transformation_create(&calibration);

	// get the new (X,Y) dimensions (as depth sensor has smaller resolution, it will be extrapolated)
	width = k4a_image_get_width_pixels(this->color);
	height = k4a_image_get_height_pixels(this->color);
	std::cout << width << ", " << height << std::endl;

	// create the depth buffer with corrected dimension
	this->result = k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16, width, height, width * (int)sizeof(uint16_t), &this->depth_colorpov);
	if (K4A_RESULT_SUCCEEDED != this->result) {
		printf("Failed to create transformed depth image\n");
		error_code = 1;
		goto Exit;
	}
	// get depth formatted to color camera dimension
	result = k4a_transformation_depth_image_to_color_camera(transformation_handle, this->depth, this->depth_colorpov);
	if (K4A_RESULT_SUCCEEDED != result) {
		printf("Failed to compute transformed depth image\n");
		error_code = 1;
		goto Exit;
	}

Exit:
	if (transformation_handle != nullptr) {
		k4a_transformation_destroy(transformation_handle);
	}
	return error_code;
}


int CaptureManager::getDepth_xyz_colorCameraPov() {
	int error_code = 0;
	k4a_calibration_t calibration;
	k4a_transformation_t transformation_handle = nullptr;
	int width, height;

	// make sure playback is available
	if (this->playback == nullptr) {
		printf("playback not found: initialise() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure color is available
	if (this->color == NULL) {
		printf("color image not found: loadColor() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure depth_colorpov is available
	if (this->depth_colorpov == NULL) {
		printf("depth colorpov image not found: getDepth_colorCameraPov() first.\n");
		error_code = 1;
		goto Exit;
	}
	// make sure that the output buffer is free
	if (this->depth_xyz_colorpov != NULL) {
		k4a_image_release(this->depth_xyz_colorpov);
		this->depth_xyz_colorpov = NULL;  // Reset to nullptr after release
	}

	// get transformation
	this->result = k4a_playback_get_calibration(this->playback, &calibration);
	if (K4A_RESULT_SUCCEEDED != this->result) {
		printf("Failed to get calibration\n");
		error_code = 1;
		goto Exit;
	}
	transformation_handle = k4a_transformation_create(&calibration);

	// get the new (X,Y) dimensions (as depth sensor has smaller resolution, it will be extrapolated)
	width = k4a_image_get_width_pixels(this->color);
	height = k4a_image_get_height_pixels(this->color);
	std::cout << width << ", " << height << std::endl;

	result = k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, width, height, 3 * width * (int)sizeof(int16_t), &this->depth_xyz_colorpov);
	if (K4A_RESULT_SUCCEEDED != result)
	{
		printf("Failed to create point cloud image\n");
		error_code = 1;
		goto Exit;
	}
	result = k4a_transformation_depth_image_to_point_cloud(transformation_handle, this->depth_colorpov, K4A_CALIBRATION_TYPE_COLOR, this->depth_xyz_colorpov);
	if (K4A_RESULT_SUCCEEDED != result) {
		printf("Failed to compute point cloud\n");
		error_code = 1;
		goto Exit;
	}

Exit:
	if (transformation_handle != nullptr) {
		k4a_transformation_destroy(transformation_handle);
	}
	return error_code;
}


int CaptureManager::generate_pointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud) {
	int error_code = 0;

	int16_t* depth_xyz_data;
	uint8_t* color_data;
	int width, height;
	int16_t x, y, z;

	// make sure depth_xyz_colorpov is available
	if (this->depth_xyz_colorpov == NULL) {
		printf("depth_xyz_colorpov image not found: getDepth_xyz_colorCameraPov() first.\n");
		error_code = 1;
		goto Exit;
	}

	// make sure color is available
	if (this->color == NULL) {
		printf("color image not found: getColor() first.\n");
		error_code = 1;
		goto Exit;
	}

	// load data
	depth_xyz_data = (int16_t*)(void*)k4a_image_get_buffer(this->depth_xyz_colorpov);
	color_data = k4a_image_get_buffer(this->color);
	width = k4a_image_get_width_pixels(this->color);
	height = k4a_image_get_height_pixels(this->color);

	// initialise point cloud with PCL library 
	if (pointCloud == nullptr) {
		pointCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	}
	pointCloud->width = width;
	pointCloud->height = height;

	// Transform the 2D point cloud into a 1D vector
	pointCloud->points.resize(pointCloud->width * pointCloud->height);
	for (size_t i = 0; i < pointCloud->points.size(); ++i) {
		x = depth_xyz_data[3 * i + 0];
		y = depth_xyz_data[3 * i + 1];
		z = depth_xyz_data[3 * i + 2];
		if (x != 0 || y != 0 || z != 0) {
			pointCloud->points[i].x = x;
			pointCloud->points[i].y = y;
			pointCloud->points[i].z = z;

			pointCloud->points[i].b = color_data[4 * i + 0];
			pointCloud->points[i].g = color_data[4 * i + 1];
			pointCloud->points[i].r = color_data[4 * i + 2];
		}
	}

Exit:
	return error_code;
}


int CaptureManager::convert_XYcoordinate_to_XYZmm(int xy[2], float xyz[3]) {
	int error_code = 0;
	// input variables
	k4a_calibration_t calibration;
	k4a_float2_t point2d;
	float z; 
	int width, height;
	k4a_float3_t point3d_source;
	// output variables
	k4a_float3_t point3d_output;
	int valid;

	// make sure depth_colorpov is available
	if (this->depth_colorpov == NULL) {
		printf("depth colorpov image not found: getDepth_colorCameraPov() first.\n");
		error_code = 1;
		goto Exit;
	}

	// make sure playback is available
	if (this->playback == nullptr) {
		printf("playback not found: initialise() first.\n");
		error_code = 1;
		goto Exit;
	}

	// get transformation
	this->result = k4a_playback_get_calibration(this->playback, &calibration);
	if (K4A_RESULT_SUCCEEDED != this->result) {
		printf("Failed to get calibration\n");
		error_code = 1;
		goto Exit;
	}

	// format xy for k4a_calibration_2d_to_3d
	point2d = { static_cast<float>(xy[0]), static_cast<float>(xy[1]) };
	width = k4a_image_get_width_pixels(this->depth_colorpov);
	height = k4a_image_get_height_pixels(this->depth_colorpov);

	// check for out of bounds
	if (point2d.xy.x < 0 || width <= point2d.xy.x) {
		std::cerr << "X value (" << point2d.xy.x << ") is out of bounds [0, " << width - 1 << "].\n";
		error_code = 1;
		goto Exit;
	}
	if (point2d.xy.y < 0 || height <= point2d.xy.y) {
		std::cerr << "Y value (" << point2d.xy.y << ") is out of bounds [0, " << height - 1 << "].\n";
		error_code = 1;
		goto Exit;
	}

	// extract z value of the neigbhourest XY of point2d
	point3d_source = findNonZeroZ(point2d);
	point2d.xy.x = point3d_source.xyz.x;
	point2d.xy.y = point3d_source.xyz.y;
	z = point3d_source.xyz.z;

	std::cout << "convert_XYcoordinate_to_XYZmm: 2D source coordinate (+ Z value at this location): (" << point2d.xy.x << ", " << point2d.xy.y << ", " << z << ")\n";
	// for Box Crop Filter: Transform XY pixel space (color image rectangle) into 3D point cloud space (pcl rectangular cuboid)
	//k4a_calibration_2d_to_3d(&calibration, &point2d_start, zs[0], K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, &point3d_start, &valid);
	this->result = k4a_calibration_2d_to_3d(&calibration, &point2d, z, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &point3d_output, &valid);
	if (K4A_RESULT_SUCCEEDED != this->result) {
		printf("Failed to k4a_calibration_2d_to_3d\n");
		error_code = 1;
		goto Exit;
	}
	if (!valid) {
		std::cerr << "convert_XYcoordinate_to_XYZmm: Failed to convert 2D point to 3D point for start\n";
		std::cerr << "Most likely, there is no depth information at the required coordinate: ("  << xy[0] << ", " << xy[1] << ")\n";
		xyz[0] = 0;
		xyz[1] = 0;
		xyz[2] = 0;
		error_code = 2;
		goto Exit;
	}
	else {
		std::cout << "-- new coordinate (3D): (" << point3d_output.xyz.x << ", " << point3d_output.xyz.y << ", " << point3d_output.xyz.z << ")\n";
		xyz[0] = point3d_output.xyz.x;
		xyz[1] = point3d_output.xyz.y;
		xyz[2] = point3d_output.xyz.z;
	}

Exit:
	return error_code;
}



k4a_float3_t CaptureManager::findNonZeroZ(const k4a_float2_t& point2d) {
	k4a_float3_t point3d_output;
	int width = k4a_image_get_width_pixels(this->depth_colorpov);
	int height = k4a_image_get_height_pixels(this->depth_colorpov);
	uint16_t* depth_buffer = reinterpret_cast<uint16_t*>(k4a_image_get_buffer(this->depth_colorpov));

	int x = static_cast<int>(point2d.xy.x);
	int y = static_cast<int>(point2d.xy.y);
	int z_idx = y * width + x;
	float z = depth_buffer[z_idx];

	if (z != 0) {
		point3d_output.xyz.x = x;
		point3d_output.xyz.y = y;
		point3d_output.xyz.z = z;
		return point3d_output;
	}

	int dx[] = { 0, 1, 0, -1 };
	int dy[] = { -1, 0, 1, 0 };
	int step = 1;
	int direction = 0;

	while (step < std::max(width, height)) {
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < step; ++j) {
				x += dx[direction];
				y += dy[direction];

				if (x >= 0 && x < width && y >= 0 && y < height) {
					z_idx = y * width + x;
					z = depth_buffer[z_idx];
					if (z != 0) {
						point3d_output.xyz.x = x;
						point3d_output.xyz.y = y;
						point3d_output.xyz.z = z;
						return point3d_output;
					}
				}
			}
			direction = (direction + 1) % 4;
		}
		++step;
	}

	point3d_output.xyz.x = -1;
	point3d_output.xyz.y = -1;
	point3d_output.xyz.z = 0;
	return point3d_output;
}



int CaptureManager::plotColor(bool blocking) {
	uint8_t* buffer = k4a_image_get_buffer(this->color);
	int width_npixels = k4a_image_get_width_pixels(this->color);
	int height_npixels = k4a_image_get_height_pixels(this->color);
	this->plotImage(buffer, width_npixels, height_npixels, CV_8UC4, "kinect color frame", blocking);
	return 0;
}

int CaptureManager::plotDepth(bool blocking) {
	uint16_t* buffer = (uint16_t*)(void*)k4a_image_get_buffer(this->depth);
	int width_npixels = k4a_image_get_width_pixels(this->depth);
	int height_npixels = k4a_image_get_height_pixels(this->depth);
	std::vector<uint8_t> rgbImage(width_npixels * height_npixels * 3);
	this->imageGrayscaleToRGB(buffer, rgbImage.data(), width_npixels, height_npixels);
	this->plotImage(rgbImage.data(), width_npixels, height_npixels, CV_8UC3, "kinect depth frame", blocking);
	return 0;
}

int CaptureManager::plotDepth_colorpov(bool blocking) {
	int width_npixels = k4a_image_get_width_pixels(this->depth_colorpov);
	int height_npixels = k4a_image_get_height_pixels(this->depth_colorpov);
	std::vector<uint8_t> rgbImage;
	this->getDepth_as_rgb(rgbImage);
	this->plotImage(rgbImage.data(), width_npixels, height_npixels, CV_8UC3, "kinect depth frame with color camera resolution", blocking);
	return 0;
}

void CaptureManager::plotImage(void* buffer, int width, int height, int type, std::string title, bool blocking) {
	cv::Mat m(height, width, type, buffer, cv::Mat::AUTO_STEP);
	cv::imshow(title, m);
	if (blocking)
		cv::waitKey(0); // Wait for a key press to close the window
}

void CaptureManager::getDepth_as_rgb(std::vector<uint8_t>& rgbImage) {
	uint16_t* buffer = (uint16_t*)(void*)k4a_image_get_buffer(this->depth_colorpov);
	int width_npixels = k4a_image_get_width_pixels(this->depth_colorpov);
	int height_npixels = k4a_image_get_height_pixels(this->depth_colorpov);
	rgbImage = std::vector<uint8_t>(width_npixels * height_npixels * 3);
	this->imageGrayscaleToRGB(buffer, rgbImage.data(), width_npixels, height_npixels);
}

// private methods
// any necessary cleanup (freeing memory)
void CaptureManager::cleanup() {
	if (this->playback != nullptr) {
		k4a_playback_close(this->playback);
	}
	if (this->capture != nullptr) {
		k4a_capture_release(this->capture);
	}
    if (this->color != NULL) {
        k4a_image_release(this->color);
    }
	if (this->depth != NULL) {
		k4a_image_release(this->depth);
	}
	if (this->depth_colorpov != NULL) {
		k4a_image_release(this->depth_colorpov);
	}
	if (this->depth_xyz_colorpov != NULL) {
		k4a_image_release(this->depth_xyz_colorpov);
	}
}

// Function to convert grayscale array to RGB array
void CaptureManager::imageGrayscaleToRGB(const uint16_t* gray, uint8_t* rgb, int width, int height) {
	int numPixels = width * height;
	float r, g, b;

	uint16_t max_gray = 0;
	for (int i = 0; i < numPixels; ++i) {
		if (max_gray < gray[i]) {
			max_gray = gray[i];
		}
	}

	for (int i = 0; i < numPixels; ++i) {
		// Read grayscale value
		float hue_like = 360 * gray[i]/(float)max_gray;
		if (hue_like > 0) {
			this->huetoRGB(hue_like, r, g, b);
			// Write RGB values (R, G, B) for the pixel
			r = r * 255;
			g = g * 255;
			b = b * 255;
			rgb[i * 3] = (uint8_t)b;        // B
			rgb[i * 3 + 1] = (uint8_t)g;    // G
			rgb[i * 3 + 2] = (uint8_t)r;    // R
		}
		else {
			// Write RGB values (R, G, B) for the pixel
			rgb[i * 3] = 0;        // B
			rgb[i * 3 + 1] = 0;    // G
			rgb[i * 3 + 2] = 0;    // R
		}
	}
}


/*! modified from https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72
Convert Hue to RGB color space
  @input:
  \param fH Hue component, range: [0, 360]
  @output:
  \param fR Red component, range: [0, 1]
  \param fG Green component,range: [0, 1]
  \param fB Blue component, [0, 1]
*/
void CaptureManager::huetoRGB(float& fH, float& fR, float& fG, float& fB) {
	float fS = 1.0;
	float fV = 1.0; 
	
	float fC = fS * fV; // Chroma
	float fHPrime = fmod(fH / 60.0, 6);
	float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
	float fM = fV - fC;

	if (0 <= fHPrime && fHPrime < 1) {
		fR = fC;
		fG = fX;
		fB = 0;
	}
	else if (1 <= fHPrime && fHPrime < 2) {
		fR = fX;
		fG = fC;
		fB = 0;
	}
	else if (2 <= fHPrime && fHPrime < 3) {
		fR = 0;
		fG = fC;
		fB = fX;
	}
	else if (3 <= fHPrime && fHPrime < 4) {
		fR = 0;
		fG = fX;
		fB = fC;
	}
	else if (4 <= fHPrime && fHPrime < 5) {
		fR = fX;
		fG = 0;
		fB = fC;
	}
	else if (5 <= fHPrime && fHPrime < 6) {
		fR = fC;
		fG = 0;
		fB = fX;
	}
	else {
		fR = 0;
		fG = 0;
		fB = 0;
	}

	fR += fM;
	fG += fM;
	fB += fM;
}
