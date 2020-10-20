#pragma once

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace torch;

class PyTorchHelper
{
private:
	torch::jit::script::Module MyNet;
	at::Device device = at::kCPU;

public:
	PyTorchHelper(string fPath, bool bGPU);
	std::vector<cv::Mat> PredictFractionalBlocks(const cv::Mat& img);

	template<typename ... Args>
	std::string format_string(const std::string& format, Args ... args)
	{
		size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1;
		std::unique_ptr<char[]> buffer(new char[size]);
		snprintf(buffer.get(), size, format.c_str(), args ...);
		return std::string(buffer.get(), buffer.get() + size - 1);
	}
};

