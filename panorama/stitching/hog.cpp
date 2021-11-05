#include "hog.hpp"

using namespace std;
using namespace cv;
using namespace stitching;

vector<HOGPair> HOG::GetHOG(const Mat& image, Moravec& moravec)
{
	Mat _image = image.clone();

	Mat img_gx; Mat img_gy;
	Sobel(_image, img_gx, CV_64FC1, 1, 0, 3);
	Sobel(_image, img_gy, CV_64FC1, 0, 1, 3);

	Mat magnitude;	Mat angle;
	// magnitude, angle 추출.
	cartToPolar(img_gx, img_gy, magnitude, angle, 1);	

	vector<HOGPair> image_histogram;

	int kernelSize = 13;
	int halfKernelSize = kernelSize / 2;

	for (int idx = 0; idx < moravec.keyPointVec.size(); idx++)
	{
		cv::Point keypoint = moravec.keyPointVec[idx];

		if ((keypoint.y - halfKernelSize >= 0) && (keypoint.x - halfKernelSize >= 0)
			&& (keypoint.y + halfKernelSize <= image.cols - 1) && (keypoint.x + halfKernelSize <= image.rows - 1))
		{
			Mat mask_Mag = magnitude(cv::Rect(keypoint.x - halfKernelSize, keypoint.y - halfKernelSize, kernelSize, kernelSize)).clone();
			Mat mask_Ang = angle(cv::Rect(keypoint.x - halfKernelSize, keypoint.y - halfKernelSize, kernelSize, kernelSize)).clone();

			// bin 9인 histogram 쌓기.
			vector<double> histogram = StackHistogram(mask_Mag, mask_Ang);	
			image_histogram.push_back(HOGPair(histogram, keypoint));
		}
	}
	return image_histogram;
}

vector<double> HOG::StackHistogram(const Mat& magnitude, const Mat& angle)
{

	vector<double> histogram(8);

	for (int y = 0; y < magnitude.rows; y++)
	{
		for (int x = 0; x < magnitude.cols; x++)
		{
			double mag_px = magnitude.at<double>(y, x);
			double ang_px = angle.at<double>(y, x);

			int idx = (ang_px / 45);

			histogram[idx] += mag_px;
		}
	}

	return histogram;
}