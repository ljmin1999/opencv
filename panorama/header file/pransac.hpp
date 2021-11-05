#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

namespace stitching
{
	// firstImgPtr에는 첫번째 이미지의 좌표가,
	// secondImgPtr에는 두번째 이미지의 좌표가 담기게 된다.
	// 두 이미지간의 Euclidean Distance를 distance에 담는다.
	struct PointSet
	{
		cv::Point firstImgPtr;
		cv::Point secondImgPtr;
		double distance;

		// PointSet간 비교를 위한 연산자 정의
		bool operator==(const PointSet& other) const {
			return firstImgPtr == other.firstImgPtr &&
				secondImgPtr == other.secondImgPtr &&
				distance == other.distance;
		}

		// PointSet간 비교를 위한 연산자 정의
		bool operator!=(const PointSet& other) const {
			return !(*this == other);
		}
	};

	struct EquationElement
	{
		double m; // 기울기
		double b; // 절편
	};

	class LeastSquare
	{
	private:
		std::random_device rn;
		std::mt19937_64 rnd;

	public:
		std::vector<PointSet> distance_pair;  // 랜덤으로 생성된 점들을 저정한 벡터

		LeastSquare() : rn(), rnd(rn()), distance_pair() { }
		int GenerateRandomNumber(int begin, int end);
		void DrawPoint(cv::InputOutputArray& first_canvas, cv::InputOutputArray& second_canvas);
		void DrawLine(EquationElement element, cv::InputOutputArray& canvas);
	};

	class PRANSAC : public LeastSquare
	{
	private:
		EquationElement lineEquation;
		EquationElement optimalEquation;

	public:
		cv::Mat T; // best homogeneous rotation mat
		int maxInlier;

		PRANSAC(std::vector<PointSet> distance_pair) : lineEquation(), optimalEquation(), maxInlier(0) {
			this->distance_pair.assign(distance_pair.begin(), distance_pair.end());
			memset(&lineEquation, 0, sizeof(EquationElement));
			memset(&optimalEquation, 0, sizeof(EquationElement));
		}

		void GetBestRotationMat(double minimum_error = 1.0);
		double GetError(const cv::InputArray& homogeneous_matrix, const PointSet& pair);
		void GetHomogeneousMatrix(const std::vector<PointSet> &selected_points, const cv::OutputArray& homogeneous_matrix);
		void AttachImage(const cv::InputArray& first_image, const cv::InputArray& second_image, const cv::OutputArray& output_image);
	};
}