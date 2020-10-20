#include "PyTorchHelper.h"

PyTorchHelper::PyTorchHelper(string fPath, bool bGPU)
{
    try {
        std::ifstream ifs(fPath, std::ifstream::binary);
        if (torch::cuda::is_available() && bGPU) {
            device = at::kCUDA;
        }
        else {
            device = at::kCPU;
        }
        MyNet = torch::jit::load(ifs, device);
        MyNet.eval();
    }
    catch (const c10::Error&) {
        std::cerr << "error loading the model\n";
    }
}

std::vector<cv::Mat> PyTorchHelper::PredictFractionalBlocks(const cv::Mat& img)
{
    int img_size = img.rows * img.cols;

    //cv::imwrite("test.png", img);

    // 입력 벡터
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC1);

    torch::Tensor img_tensor = torch::from_blob(float_img.data, { 1, img.rows, img.cols, 1 }, torch::kFloat).to(device);
    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(img_tensor);

    // 예측
    torch::Tensor netOutput = MyNet.forward(input_tensors).toTensor();
    netOutput = netOutput.permute({ 0, 2, 3, 1 });
    // std::cout << netOutput.sizes() << std::endl;

    // 재배치
    auto output_tensor = netOutput.cpu().detach();
    
    // 결과값 크기
    size_t output_size = output_tensor.numel();

    /*float* dOutput = (float*)malloc(sizeof(float) * output_size);
    std::memcpy(dOutput, output_tensor.data_ptr(), sizeof(float) * output_size);*/

    /*int _sz[] = { img.rows, img.cols, 15 };
    cv::Mat result_img(3, _sz, CV_32FC1);
    std::memcpy((void*)result_img.data, output_tensor.data_ptr(), sizeof(float) * output_size);*/

    std::vector<cv::Mat> preds;
    //cv::split(result_img, preds);
    int num_img = netOutput.sizes()[3];
    for (int i = 0; i < num_img; i++) {
        cv::Mat result_img(img.rows, img.cols, CV_32FC1);
        std::memcpy((void*)result_img.data, (float*)output_tensor.data_ptr() + (img_size * i), sizeof(float) * img_size);
        //cv::Mat uchar_img;
        //result_img.convertTo(uchar_img, CV_16SC1);
        preds.push_back(result_img);

        //cv::imwrite(format_string("%s%d%s", "test", i, ".png"), result_img);
    }
    
    return preds;
    // return std::vector<cv::Mat>();
}
