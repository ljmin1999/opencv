#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

namespace stitching
{
	// firstImgPtr���� ù��° �̹����� ��ǥ��,
	// secondImgPtr���� �ι�° �̹����� ��ǥ�� ���� �ȴ�.
	// �� �̹������� Euclidean Distance�� distance�� ��´�.
	struct PointSet
	{
		cv::Point firstImgPtr;
		cv::Point secondImgPtr;
		double distance;

		// PointSet�� �񱳸� ���� ������ ����
		bool operator==(const PointSet& other) const {
			return firstImgPtr == other.firstImgPtr &&
				secondImgPtr == other.secondImgPtr &&
				distance == other.distance;
		}

		// PointSet�� �񱳸� ���� ������ ����
		bool operator!=(const PointSet& other) const {
			return !(*this == other);
		}
	};

	struct EquationElement
	{
		double m; // ����
		double b; // ����
	};

	class LeastSquare
	{
	private:
		std::random_device rn;
		std::mt19937_64 rnd;

	public:
		std::vector<PointSet> distance_pair;  // �������� ������ ������ ������ ����

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