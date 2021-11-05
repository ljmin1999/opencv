#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "moravec.hpp"

namespace stitching
{
	class HOGPair
	{
	public:
		std::vector<double> hog;
		cv::Point keypoint;
		HOGPair(std::vector<double> HOG, cv::Point pt) : hog(std::move(HOG)), keypoint(pt) {}
	};

	class HOG
	{
	public:
		std::vector<HOGPair> GetHOG(const cv::Mat& image, Moravec& moravec);
		std::vector<double> StackHistogram(const cv::Mat& magnitude, const cv::Mat& angle);
	};
}