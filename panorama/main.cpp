#include <chrono>
#include <cstdlib>
#include <ctime>

#include "opencv2/opencv.hpp"
#include "stitching/pransac.hpp"
#include "stitching/moravec.hpp"
#include "stitching/hog.hpp"
#include "stitching/distance_metric.hpp"

using namespace stitching;

bool sort_by_distance_ascending(PointSet& a, PointSet& b)
{
	return b.distance > a.distance;
}

bool compare_ptr_loosen(const cv::Point& a, const cv::Point& b, const double loosenPx = 2)
{
	if (abs(a.x - b.x) < loosenPx && abs(a.y - b.y) < loosenPx)
		return true;
	return false;
}

int main()
{
	std::srand(static_cast<unsigned int>(std::time(0)));

	cv::Mat first_image_color, second_image_color, first_image, second_image;
	cv::Mat first_image_gaussian, second_image_gaussian;
	cv::Mat attached_image;

	first_image_color = cv::imread("landscape-left.bmp", cv::IMREAD_COLOR);
	second_image_color = cv::imread("landscape-right.bmp", cv::IMREAD_COLOR);

	cv::cvtColor(first_image_color, first_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(second_image_color, second_image, cv::COLOR_BGR2GRAY);

	assert(first_image.rows > 200 && first_image.cols > 200);
	assert(second_image.rows > 200 && second_image.cols > 200);

	/**
	 * 모라벡 알고리즘을 이용한 Keypoint 검출
	 */
	Moravec moravec_first(first_image);
	moravec_first.CreateConfidence();
	moravec_first.FindKeyPoint();

	Moravec moravec_second(second_image);
	moravec_second.CreateConfidence();
	moravec_second.FindKeyPoint();

	cv::imshow("moravec_first.bmp", moravec_first.DrawKeyPoint());
	cv::imshow("moravec_second.bmp", moravec_second.DrawKeyPoint());
	cv::waitKey();

	cv::imwrite("moravec_first.bmp", moravec_first.DrawKeyPoint());
	cv::imwrite("moravec_second.bmp", moravec_second.DrawKeyPoint());

	/**
	 * 그레이디언트의 히스토그램을 통한 특징 추출
	 */
	HOG hog;
	std::vector<HOGPair> first_image_hog = hog.GetHOG(first_image, moravec_first);
	std::vector<HOGPair> second_image_hog = hog.GetHOG(second_image, moravec_second);

	/**
	 * 추출한 특징간 거리 비교 및 두 이미지간 특징 Pair 검출
	 */
	DistanceCalculator euclidean_distance;
	euclidean_distance.ExtractMinDistance(first_image_hog, second_image_hog);
	euclidean_distance.DisplayPairLine(first_image, second_image);
	std::vector<PointSet> distance_pair = euclidean_distance.GetDistancePair();

	/**
	 * 변환 행렬 구하기 단계
	 *
	 * 구한 distance_pair를 이용하여 변환 행렬을 구한다.
	 * 교재 p.325의 알고리즘 7-9를 이용하여 기하 변환 행렬 T를 추정
	 * 
	 * 기하변환 T는 이미지#1을 이미지#2에 해당하는 점으로 변환하는 행렬
	 * 이 특성을 이용하여, 후방 기하 변환을 이용하여 새로운 이미지 생성
	 */
	PRANSAC ransac(distance_pair);
	ransac.GetBestRotationMat();
	ransac.AttachImage(first_image_color, second_image_color, attached_image);

	cv::imshow("Attached Image", attached_image);
	cv::waitKey();

	cv::imwrite("attached-landscape.bmp", attached_image);
	std::cout << "파일 저장 완료!" << std::endl;
}