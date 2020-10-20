#pragma once
#include <TLibCommon/CommonDef.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace torch;

class PyTorchModelExecutor
{
private:
	torch::jit::script::Module MyNet;
	at::Device device = at::kCPU;
	torch::NoGradGuard no_grad_;

public:
	PyTorchModelExecutor(string fpath);
	float* PredictFractionalPel(float* src, int channel, int height, int width);
	short convert_to_integer(double v);
	double convert_to_float(short v);
	void convert_to_integer(float* src, uint8_t* dst, int width, int height, int channel);
	void convert_to_float(uint8_t* src, float* dst, int width, int height, int channel);
	void ToFloat(Pel* src, float* dst, int width, int height);
	void ToInteger(float* src, Pel* dst, int width, int height);
	void WHC_to_CHW(Pel* src, uint8_t* dst, int width, int height);
	uint8_t CLAMP(int16_t value) { return value < 0 ? 0 : (value > 255 ? 255 : value); }
	
	// color space conversion
	void yuyv_to_rgb(unsigned char* yuv_buffer, unsigned char* rgb_buffer, int iWidth, int iHeight);
	void NV21_YUV420P(const unsigned char* image_src, unsigned char* image_dst, int image_width, int image_height);
	int yuyv_to_yuv420p(const unsigned char* in, unsigned char* out, unsigned int width, unsigned int height);
	void YUV420P_to_RGB24(unsigned char* data, unsigned char* rgb, int width, int height);
	void NV21_TO_RGB24(unsigned char* yuyv, unsigned char* rgb, int width, int height);
	void YUV420P_TO_RGB888(uint8_t* yuv420p, uint8_t* rgb888, int width, int height);
	void YUV420P_TO_RGB888(Pel* y, Pel* u, Pel* v, uint8_t* rgb888, int width, int height);
	void YUV420P_TO_CHW(Pel* y, Pel* u, Pel* v, uint8_t* bgr888, int width, int height);
	void CHW_TO_RGB888(uint8_t* src, uint8_t* dst, int width, int height);
	void RGB888_TO_YUV420P(int w, int h, uint8_t* rgb, uint8_t* yuv);
	void RGB888_TO_YUV420P(int w, int h, uint8_t* rgb, Pel* y, Pel* u, Pel* v);
};