#include "PyTorch.h"

/// <summary>
/// 생성자
/// </summary>
/// <param name="fpath">파이토치 학습 모델 파일 경로</param>
PyTorchModelExecutor::PyTorchModelExecutor(string fpath)
{
	try {
        std::ifstream is(fpath, std::ifstream::binary);
        if (torch::cuda::is_available()) {
            device = at::kCUDA;
        }
        MyNet = torch::jit::load(is, device);
		//MyNet = torch::jit::load(fpath, torch::kCUDA);
        
        MyNet.eval();
	}
	catch (const c10::Error&) {
		std::cerr << "error loading the model\n";
	}
}

float* PyTorchModelExecutor::PredictFractionalPel(float* src, int channel, int height, int width)
{
    
    // 입력 벡터
	std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(torch::from_blob(src, { 1, channel, height, width }, torch::kFloat).to(device));

    // std::cout << temp.sizes() << ':' << temp.dim() << '\n';

    // 예측
    torch::Tensor netOutput = MyNet.forward(input_tensors).toTensor();
    // torch::Tensor myOutput = netOutput.toTuple()->elements()[0].toTensor();
	// std::cout << netOutput.sizes() << ":" << netOutput.dim() << ":" << netOutput.dtype() << '\n';
    
    // 재배치
    //auto output_tensor = myOutput.permute({ 0, 3, 2, 1 }).cpu().detach();
    auto output_tensor = netOutput.cpu().detach();
    //auto output_tensor3 = myOutput.reshape({ 1, 256, 256, 3 }).cpu().detach();

    //auto oTemp1 = output_tensor.data_ptr();
    //auto oTemp2 = output_tensor.data_ptr();
    //auto oTemp3 = output_tensor3.data_ptr();

    // 결과값 크기
    size_t output_size = output_tensor.numel();
    //auto p = static_cast<float*>(output_tensor.storage().data());

    // 버퍼 할당
    //auto output_vector = std::vector<float>(output_size);

    //auto output_data = output_tensor.item();

    // 복사
    //for (int i = 0; i < output_size; i++)
    //{
    //    output_vector[i] = p[i];
    //}

    auto tShape = output_tensor.sizes().data();
    // int len = output_tensor.numel();
    float* dOutput = (float *)malloc(sizeof(float) * output_size);
    std::memcpy(dOutput, output_tensor.data_ptr(), sizeof(float) * output_size);

	//double* output = (double*)calloc(len, 6);;
	//// Copy into output
	
	return dOutput;
}

void PyTorchModelExecutor::WHC_to_CHW(Pel* src, uint8_t* dst, int width, int height)
{
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            dst[(i * height) + j] = src[(j * width) + i];
        }
    }
}

void PyTorchModelExecutor::ToFloat(Pel* src, float* dst, int width, int height)
{
    int len = width * height;
    for (int i = 0; i < len; i++)
    {
        float val = static_cast<float>(src[i] / 255.);
        dst[i] = val;
    }
}

void PyTorchModelExecutor::ToInteger(float* src, Pel* dst, int width, int height)
{
    int len = width * height;

    for (int i = 0; i < len; i++) {
        uint8_t val = static_cast<uint8_t>(src[i] * 255.);
        dst[i] = CLAMP(val);
    }
}

short PyTorchModelExecutor::convert_to_integer(double v)
{
	short val = (short)v * 255;
	if (val > 255)
	{
		val = 255;
	}
	return val;
}

double PyTorchModelExecutor::convert_to_float(short v)
{
	return (double)v / 255.;
}

void PyTorchModelExecutor::convert_to_integer(float* src, uint8_t* dst, int width, int height, int channel)
{
    int len = (channel == 0) ? (width * height) : (width * height * channel);
    
    for (int i = 0; i < len; i++) {
        uint8_t val = static_cast<uint8_t>(src[i] * 255.);
        dst[i] = CLAMP(val);
    }
}

void PyTorchModelExecutor::convert_to_float(uint8_t* src, float* dst, int width, int height, int channel)
{
    int len = (channel == 0) ? (width * height) : (width * height * channel);
    float val = {};
    for (int i = 0; i < len; i++) {
        val = static_cast<float>(src[i] / 255. );
        dst[i] = val;
    }
}

/*
Function: Convert YUV data to RGB format
 Function parameters:
 unsigned char *yuv_buffer: YUV source data
 unsigned char *rgb_buffer: RGB data after conversion
 int iWidth, int iHeight: the width and height of the image
*/
void PyTorchModelExecutor::yuyv_to_rgb(unsigned char* yuv_buffer, unsigned char* rgb_buffer, int iWidth, int iHeight)
{
    int x;
    int z = 0;
    unsigned char* ptr = rgb_buffer;
    unsigned char* yuyv = yuv_buffer;
    for (x = 0; x < iWidth * iHeight; x++)
    {
        int r, g, b;
        int y, u, v;

        if (!z)
            y = yuyv[0] << 8;
        else
            y = yuyv[2] << 8;
        u = yuyv[1] - 128;
        v = yuyv[3] - 128;

        r = (y + (359 * v)) >> 8;
        g = (y - (88 * u) - (183 * v)) >> 8;
        b = (y + (454 * u)) >> 8;

        *(ptr++) = (r > 255) ? 255 : ((r < 0) ? 0 : r);
        *(ptr++) = (g > 255) ? 255 : ((g < 0) ? 0 : g);
        *(ptr++) = (b > 255) ? 255 : ((b < 0) ? 0 : b);

        if (z++)
        {
            z = 0;
            yuyv += 4;
        }
    }
}



// image_src is the source image, image_dst is the converted image
void PyTorchModelExecutor::NV21_YUV420P(const unsigned char* image_src, unsigned char* image_dst, int image_width, int image_height)
{
    unsigned char* p = image_dst;
    memcpy(p, image_src, image_width * image_height * 3 / 2);
    const unsigned char* pNV = image_src + image_width * image_height;
    unsigned char* pU = p + image_width * image_height;
    unsigned char* pV = p + image_width * image_height + ((image_width * image_height) >> 2);
    for (int i = 0; i < (image_width * image_height) / 2; i++)
    {
        if ((i % 2) == 0)
            *pV++ = *(pNV + i);
        else
            *pU++ = *(pNV + i);
    }
}


//YUYV==YUV422
int PyTorchModelExecutor::yuyv_to_yuv420p(const unsigned char* in, unsigned char* out, unsigned int width, unsigned int height)
{
    unsigned char* y = out;
    unsigned char* u = out + width * height;
    unsigned char* v = out + width * height + width * height / 4;

    unsigned int i, j;
    unsigned int base_h;
    unsigned int  is_u = 1;
    unsigned int y_index = 0, u_index = 0, v_index = 0;

    unsigned long yuv422_length = 2 * width * height;

    //The sequence is YU YV YU YV, the length of a yuv422 frame width * height * 2 bytes
    //Discard even rows u v
    for (i = 0; i < yuv422_length; i += 2)
    {

        *(y + y_index) = *(in + i);

        y_index++;

    }

    for (i = 0; i < height; i += 2)
    {

        base_h = i * width * 2;

        for (j = base_h + 1; j < base_h + width * 2; j += 2)

        {

            if (is_u)
            {

                *(u + u_index) = *(in + j);

                u_index++;

                is_u = 0;

            }

            else
            {
                *(v + v_index) = *(in + j);
                v_index++;
                is_u = 1;
            }
        }
    }
    return 1;
}


void PyTorchModelExecutor::YUV420P_to_RGB24(unsigned char* data, unsigned char* rgb, int width, int height)
{
    int index = 0;
    unsigned char* ybase = data;
    unsigned char* ubase = &data[width * height];
    unsigned char* vbase = &data[width * height * 5 / 4];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //YYYYYYYYUUVV
            uint8_t Y = ybase[x + y * width];
            uint8_t U = ubase[y / 2 * width / 2 + (x / 2)];
            uint8_t V = vbase[y / 2 * width / 2 + (x / 2)];
            rgb[index++] = Y + 1.402 * (V - 128); //R
            rgb[index++] = Y - 0.34413 * (U - 128) - 0.71414 * (V - 128); //G
            rgb[index++] = Y + 1.772 * (U - 128); //B
        }
    }
}


/**
   * NV21 is the default format of android camera
 * @param data
 * @param rgb
 * @param width
 * @param height
 */
void PyTorchModelExecutor::NV21_TO_RGB24(unsigned char* yuyv, unsigned char* rgb, int width, int height)
{
    const int nv_start = width * height;
    int  index = 0, rgb_index = 0;
    uint8_t y, u, v;
    int r, g, b, nv_index = 0, i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            //nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
            nv_index = i / 2 * width + j - j % 2;

            y = yuyv[rgb_index];
            u = yuyv[nv_start + nv_index];
            v = yuyv[nv_start + nv_index + 1];

            r = y + (140 * (v - 128)) / 100;  //r
            g = y - (34 * (u - 128)) / 100 - (71 * (v - 128)) / 100; //g
            b = y + (177 * (u - 128)) / 100; //b

            if (r > 255)   r = 255;
            if (g > 255)   g = 255;
            if (b > 255)   b = 255;
            if (r < 0)     r = 0;
            if (g < 0)     g = 0;
            if (b < 0)     b = 0;

            index = rgb_index % width + (height - i - 1) * width;
            //rgb[index * 3+0] = b;
            //rgb[index * 3+1] = g;
            //rgb[index * 3+2] = r;

                             //Invert the image
            //rgb[height * width * 3 - i * width * 3 - 3 * j - 1] = b;
            //rgb[height * width * 3 - i * width * 3 - 3 * j - 2] = g;
            //rgb[height * width * 3 - i * width * 3 - 3 * j - 3] = r;

                             //Front image
            rgb[i * width * 3 + 3 * j + 0] = b;
            rgb[i * width * 3 + 3 * j + 1] = g;
            rgb[i * width * 3 + 3 * j + 2] = r;

            rgb_index++;
        }
    }
}

void PyTorchModelExecutor::YUV420P_TO_RGB888(uint8_t* yuv420p, uint8_t* rgb888, int width, int height)
{
    int index = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int indexY = y * width + x;
            int indexU = width * height + y / 2 * width / 2 + x / 2;
            int indexV = width * height + width * height / 4 + y / 2 * width / 2 + x / 2;

            uint8_t Y = yuv420p[indexY];
            uint8_t U = yuv420p[indexU];
            uint8_t V = yuv420p[indexV];

            rgb888[index++] = Y + 1.402 * (V - 128); //R
            rgb888[index++] = Y - 0.34413 * (U - 128) - 0.71414 * (V - 128); //G
            rgb888[index++] = Y + 1.772 * (U - 128); //B
        }
    }
}

void PyTorchModelExecutor::YUV420P_TO_RGB888(Pel* y, Pel* u, Pel* v, uint8_t* rgb888, int width, int height)
{
    int r, g, b;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int yy = y[(j * width) + i];
            int uu = u[((j / 2) * (width / 2)) + (i / 2)];
            int vv = v[((j / 2) * (width / 2)) + (i / 2)];

            r = yy + 1.402      * (vv - 128);                           //R
            g = yy - 0.34413    * (uu - 128) - 0.71414 * (vv - 128);    //G
            b = yy + 1.772      * (uu - 128);                           //B

            *rgb888++ = CLAMP(r);
            *rgb888++ = CLAMP(g);
            *rgb888++ = CLAMP(b);
        }
    }
}

void PyTorchModelExecutor::YUV420P_TO_CHW(Pel* y, Pel* u, Pel* v, uint8_t* bgr888, int width, int height)
{
    int r, g, b;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int yy = y[(j * width) + i];
            int uu = u[((j / 2) * (width / 2)) + (i / 2)];
            int vv = v[((j / 2) * (width / 2)) + (i / 2)];

            r = yy + 1.402 * (vv - 128); //R
            g = yy - 0.34413 * (uu - 128) - 0.71414 * (vv - 128); //G
            b = yy + 1.772 * (uu - 128); //B

            bgr888[(j * width) + i]                           = CLAMP(b);
            bgr888[(j * width) + i + (width * height)]        = CLAMP(g);
            bgr888[(j * width) + i + (width * height * 2)]    = CLAMP(r);
        }
    }
}

void PyTorchModelExecutor::CHW_TO_RGB888(uint8_t* src, uint8_t* dst, int width, int height)
{
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            *dst++ = src[(j * width) + i + (width * height * 2)];
            *dst++ = src[(j * width) + i + (width * height * 1)];
            *dst++ = src[(j * width) + i];
        }
    }
}

void PyTorchModelExecutor::RGB888_TO_YUV420P(int w, int h, uint8_t* rgb, uint8_t* yuv)
{
    int pixsize;
    int pixIndex;
    uint8_t* y, * u, * v;
    int i, j;

    pixsize = w * h; //image size
    y = yuv; //yuv420P y start address
    u = yuv + pixsize; //yuv420P u get the starting address
    v = u + pixsize / 4; //yuv420P v start address

    for (i = 0; i < h; i++) {

        for (j = 0; j < w; j++) {
            pixIndex = i * 3 * w + j * 3;
            int nr = rgb[pixIndex]; //Get the r, g, b value of each pixel
            int ng = rgb[pixIndex + 1];
            int nb = rgb[pixIndex + 2];

            *y++ = (uint8_t)((66 * nr + 129 * ng + 25 * nb + 128) >> 8) + 16;

            if ((i % 2 == 1) && (j % 2 == 1)) {
                *u++ = (uint8_t)((-38 * nr - 74 * ng + 112 * nb + 128) >> 8) + 128;
                *v++ = (uint8_t)((112 * nr - 94 * ng - 18 * nb + 128) >> 8) + 128;
            }
        }
    }

    return;
}

void PyTorchModelExecutor::RGB888_TO_YUV420P(int w, int h, uint8_t* rgb, Pel* y, Pel* u, Pel* v)
{
    int pixsize;
    int pixIndex;
    int i, j;

    pixsize = w * h; //image size

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            pixIndex = i * 3 * w + j * 3;
            int nr = rgb[pixIndex    ]; // Get the r, g, b value of each pixel
            int ng = rgb[pixIndex + 1];
            int nb = rgb[pixIndex + 2];

            *y++ = (uint8_t)((66 * nr + 129 * ng + 25 * nb + 128) >> 8) + 16;

            if ((i % 2 == 1) && (j % 2 == 1)) 
            {
                *u++ = (uint8_t)((-38 * nr - 74 * ng + 112 * nb + 128) >> 8) + 128;
                *v++ = (uint8_t)((112 * nr - 94 * ng - 18 * nb + 128) >> 8) + 128;
            }
        }
    }

    return;
}