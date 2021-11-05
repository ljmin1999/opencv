#include "moravec.hpp"

using namespace stitching;

Moravec::Moravec(const cv::Mat& image)
{
	image.copyTo(this->image);
	confidence_map.create(image.size(), CV_32FC1);
	confidence_map.setTo(cv::Scalar(.0f));
}

void Moravec::CreateConfidence()
{
	for (int h = 2; h < image.rows - 2; ++h)
		for (int w = 2; w < image.cols - 2; ++w)
		{
			// 3x3 그리드의 중점 (0, 0)
			// S1 = (-1, 0), S2 = (1, 0), S3 = (0, -1), S4 = (0, 1)
			int s1 = 0;			int s2 = 0;
			int s3 = 0;			int s4 = 0;

			// 3x3 그리드
			for (int y = -1; y <= 1; ++y)
				for (int x = -1; x <= 1; ++x)
				{
					// 제곱차합을 계산
					float M = image.at<uchar>(h + y, w + x);
					float top_diff = M - image.at<uchar>(h + y - 1, w + x);// 상
					float bottom_diff = M - image.at<uchar>(h + y + 1, w + x);// 하
					float left_diff = M - image.at<uchar>(h + y, w + x - 1);// 좌
					float right_diff = M - image.at<uchar>(h + y, w + x + 1);// 우

					s1 += pow(top_diff, 2);
					s2 += pow(bottom_diff, 2);
					s3 += pow(left_diff, 2);
					s4 += pow(right_diff, 2);
				}

			// 가장 작은 값을 confidence_map에 추가
			int c = MIN(MIN(MIN(s1, s2), s3), s4);

			confidence_map.at<float>(h, w) = c;
		}
}

void Moravec::FindKeyPoint()
{
	for (int y = 16; y < image.rows - 16; ++y)
		for (int x = 16; x < image.cols - 16; ++x)
		{
			// confidence_map을 순회하면서 임계값보다 값이 크면
			// 해당 값이 위치한 좌표를 keyPointVec에 추가
			if (confidence_map.at<float>(y, x) > threshold)
				keyPointVec.push_back(cv::Point(x, y));
		}
}

cv::Mat Moravec::DrawKeyPoint()
{
	cv::Mat canvas;
	cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < keyPointVec.size(); i++)
	{
		cv::Point keyPoint = keyPointVec[i];
		cv::circle(canvas, cv::Point(keyPoint.x, keyPoint.y), 3, cv::Scalar(0, 0, 255));
	}

	return canvas;
}
