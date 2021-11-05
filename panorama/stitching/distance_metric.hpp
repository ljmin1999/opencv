#pragma once
#include <opencv2/opencv.hpp>
#include "moravec.hpp"
#include "hog.hpp"
#include "pransac.hpp"

namespace stitching
{
	class DistanceCalculator
	{
	private:
		std::vector<PointSet> distance_pair;

	public:
		void ExtractMinDistance(const std::vector<HOGPair>& h1, const std::vector<HOGPair>& h2);
		double CalculateDistance(const std::vector<double>& h1, const std::vector<double>& h2);
		void DisplayPairLine(const cv::Mat& image1, const cv::Mat& image2);

		std::vector<PointSet> GetDistancePair()
		{
			return distance_pair;
		}
	};
}
