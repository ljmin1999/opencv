#pragma once
#include <opencv2/opencv.hpp>

namespace stitching
{
	class Moravec
	{
	private:
		cv::Mat image;
		const int threshold = 25000;
		cv::Mat confidence_map;

	public:
		std::vector<cv::Point> keyPointVec;

		Moravec(const cv::Mat& image);
		void CreateConfidence();
		void FindKeyPoint();
		cv::Mat DrawKeyPoint();

	};
}