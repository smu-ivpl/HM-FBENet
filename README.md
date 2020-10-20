# HEVC Model - Fractional Block Enhancement Network
Modified HEVC Reference Software (HM-16.15) for Fractional Block Enhancement Experiment in Inter-Coding Stage

## Pre-requisites
- Visual Studio 2019
- C++-14
- Windows SDK 10.0.19041.0
- LibTorch 1.6.0
- CUDA 10.1
- CuDNN 8.0.3

## VS2019 Solution Settings

CUDA, LibTorch, OpenCV 경로 설정 예시 (알맞게 수정할 것)

Platform - x64

- TAppEncoder 프로젝트에서
    - 작업디렉터리 설정
        - Debugging - Working Directory
            - ../../working

- 모든 프로젝트에서
    - C/C++ - General - Additional Include Directories **(헤더 파일 경로)**
    
        - Configuration - All Configuration
        
            - D:\Libraries\opencv\include
            - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include
            - C:\Program Files\NVIDIA Corporation\NvToolsExt\include
            
        - Configuration - Debug
        
            - D:\Libraries\LibTorch\libtorch160_cu101_debug\include\torch\csrc\api\include
            - D:\Libraries\LibTorch\libtorch160_cu101_debug\include
            
        - Configuration - Release
        
            - D:\Libraries\LibTorch\libtorch160_cu101_release\include\torch\csrc\api\include
            - D:\Libraries\LibTorch\libtorch160_cu101_release\include

- TAppEncoder 프로젝트에서 **(TLib* 프로젝트들에는 링커 설정이 없음)**

    - Debugging - Environment **(.dll 파일 경로)**
    
        - Configuration - All Configuration
        
            - PATH=
            D:\Libraries\opencv\build\x64\vc15\bin;
            C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64;
            C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;
            %PATH%
            
        - Configuration - Debug
        
            - PATH=
            D:\Libraries\LibTorch\libtorch160_cu101_debug\lib;
            %PATH%
            
        - Configuration - Release
        
            - PATH=
            D:\Libraries\LibTorch\libtorch160_cu101_release\lib;
            %PATH%
            
    - Linker - General - Additional Library Directories **(.lib 파일 경로)**
    
        - Configuration - All Configuration
        
            - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64
            - C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64
            - D:\Libraries\opencv\build\x64\vc15\lib
            
        - Configuration - Debug
        
            - D:\Libraries\LibTorch\libtorch160_cu101_debug\lib
            
        - Configuration - Release
        
            - D:\Libraries\LibTorch\libtorch160_cu101_release\lib
            
    - Linker - Input - Additional Dependencies **(.lib 파일 이름)**
    
        - Configuration - All Configuration

            c10.lib
            caffe2_nvrtc.lib
            c10_cuda.lib
            torch.lib
            torch_cuda.lib
            torch_cpu.lib
            -INCLUDE:?warp_size@cuda@at@@YAHXZ
            nvToolsExt64_1.lib
            cudart_static.lib
            cufft.lib
            curand.lib
            cublas.lib
            cudnn.lib

        - Configuration - Debug

            opencv_world440**d**.lib

        - Configuration - Release

            opencv_world440.lib
            
## 주요 소스
``Project - TLibCommon``
- PyTorchHelper   {.h, .cpp}  : 파이토치 래퍼
- TComPicYuv      {.h, .cpp}  : YUV 버퍼 클래스, 파이토치 기반 분수화소 생성 함수

``Project - TLibEncoder``
- TEncSearch      {.h, .cpp}  : 하프-펠, 쿼터-펠 생성 함수 (b_ExpCNN 플래그 값으로 CNN 분수화소 생성 모드를 ON/OFF)
