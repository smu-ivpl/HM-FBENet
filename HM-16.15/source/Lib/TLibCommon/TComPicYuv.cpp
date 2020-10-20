/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPicYuv.cpp
    \brief    picture YUV buffer class
*/

#include <cstdlib>
#include <assert.h>
#include <memory.h>

#ifdef __APPLE__
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include "TComPicYuv.h"
#include "TLibVideoIO/TVideoIOYuv.h"

#include "PyTorchHelper.h"
#include "cuda_runtime.h"

//! \ingroup TLibCommon
//! \{

TComPicYuv::TComPicYuv()
{
  for(UInt i=0; i<MAX_NUM_COMPONENT; i++)
  {
    m_apiPicBuf[i]    = NULL;   // Buffer (including margin)
    m_piPicOrg[i]     = NULL;    // m_apiPicBufY + m_iMarginLuma*getStride() + m_iMarginLuma
  }
  m_PicFilteredByCNN = NULL;

  for(UInt i=0; i<MAX_NUM_CHANNEL_TYPE; i++)
  {
    m_ctuOffsetInBuffer[i]=0;
    m_subCuOffsetInBuffer[i]=0;
  }

  m_bIsBorderExtended = false;
  bEdited = false;

  /*model_h = new PyTorchHelper(fpath_h, bUseGPU);
  model_q = new PyTorchHelper(fpath_q, bUseGPU);*/

  /*try {
      
      if (torch::cuda::is_available() && bUseGPU) {
          device = at::kCUDA;
      }
      else {
          device = at::kCPU;
      }

      std::ifstream ifs(fpath_h, std::ifstream::binary);
      modelH = torch::jit::load(ifs, device);
      ifs.close();

      ifs = std::ifstream(fpath_q, std::ifstream::binary);
      modelQ = torch::jit::load(ifs, device);
      ifs.close();

      modelH.eval();
      modelQ.eval();
  }
  catch (const c10::Error&) {
      std::cerr << "error loading the model\n";
  }*/
}




TComPicYuv::~TComPicYuv()
{
  destroy();
}



Void TComPicYuv::createWithoutCUInfo ( const Int picWidth,                 ///< picture width
                                       const Int picHeight,                ///< picture height
                                       const ChromaFormat chromaFormatIDC, ///< chroma format
                                       const Bool bUseMargin,              ///< if true, then a margin of uiMaxCUWidth+16 and uiMaxCUHeight+16 is created around the image.
                                       const UInt maxCUWidth,              ///< used for margin only
                                       const UInt maxCUHeight)             ///< used for margin only

{
  destroy();

  m_picWidth          = picWidth;
  m_picHeight         = picHeight;
  m_chromaFormatIDC   = chromaFormatIDC;
  m_marginX          = (bUseMargin?maxCUWidth:0) + 16;   // for 16-byte alignment
  m_marginY          = (bUseMargin?maxCUHeight:0) + 16;  // margin for 8-tap filter and infinite padding
  m_bIsBorderExtended = false;

  // assign the picture arrays and set up the ptr to the top left of the original picture
  for(UInt comp=0; comp<getNumberValidComponents(); comp++)
  {
    const ComponentID ch=ComponentID(comp);
    m_apiPicBuf[comp] = (Pel*)xMalloc( Pel, getStride(ch) * getTotalHeight(ch));
    m_piPicOrg[comp]  = m_apiPicBuf[comp] + (m_marginY >> getComponentScaleY(ch)) * getStride(ch) + (m_marginX >> getComponentScaleX(ch));
  }
  // initialize pointers for unused components to NULL
  for(UInt comp=getNumberValidComponents();comp<MAX_NUM_COMPONENT; comp++)
  {
    m_apiPicBuf[comp] = NULL;
    m_piPicOrg[comp]  = NULL;
  }

  for(Int chan=0; chan<MAX_NUM_CHANNEL_TYPE; chan++)
  {
    m_ctuOffsetInBuffer[chan]   = NULL;
    m_subCuOffsetInBuffer[chan] = NULL;
  }
}



Void TComPicYuv::create ( const Int picWidth,                 ///< picture width
                          const Int picHeight,                ///< picture height
                          const ChromaFormat chromaFormatIDC, ///< chroma format
                          const UInt maxCUWidth,              ///< used for generating offsets to CUs.
                          const UInt maxCUHeight,             ///< used for generating offsets to CUs.
                          const UInt maxCUDepth,              ///< used for generating offsets to CUs.
                          const Bool bUseMargin)              ///< if true, then a margin of uiMaxCUWidth+16 and uiMaxCUHeight+16 is created around the image.

{
  createWithoutCUInfo(picWidth, picHeight, chromaFormatIDC, bUseMargin, maxCUWidth, maxCUHeight);


  const Int numCuInWidth  = m_picWidth  / maxCUWidth  + (m_picWidth  % maxCUWidth  != 0);
  const Int numCuInHeight = m_picHeight / maxCUHeight + (m_picHeight % maxCUHeight != 0);
  for(Int chan=0; chan<MAX_NUM_CHANNEL_TYPE; chan++)
  {
    const ChannelType ch= ChannelType(chan);
    const Int ctuHeight = maxCUHeight>>getChannelTypeScaleY(ch);
    const Int ctuWidth  = maxCUWidth>>getChannelTypeScaleX(ch);
    const Int stride    = getStride(ch);

    m_ctuOffsetInBuffer[chan] = new Int[numCuInWidth * numCuInHeight];

    for (Int cuRow = 0; cuRow < numCuInHeight; cuRow++)
    {
      for (Int cuCol = 0; cuCol < numCuInWidth; cuCol++)
      {
        m_ctuOffsetInBuffer[chan][cuRow * numCuInWidth + cuCol] = stride * cuRow * ctuHeight + cuCol * ctuWidth;
      }
    }

    m_subCuOffsetInBuffer[chan] = new Int[(size_t)1 << (2 * maxCUDepth)];

    const Int numSubBlockPartitions=(1<<maxCUDepth);
    const Int minSubBlockHeight    =(ctuHeight >> maxCUDepth);
    const Int minSubBlockWidth     =(ctuWidth  >> maxCUDepth);

    for (Int buRow = 0; buRow < numSubBlockPartitions; buRow++)
    {
      for (Int buCol = 0; buCol < numSubBlockPartitions; buCol++)
      {
        m_subCuOffsetInBuffer[chan][(buRow << maxCUDepth) + buCol] = stride  * buRow * minSubBlockHeight + buCol * minSubBlockWidth;
      }
    }
  }
}

Void TComPicYuv::destroy()
{
  for(Int comp=0; comp<MAX_NUM_COMPONENT; comp++)
  {
    m_piPicOrg[comp] = NULL;

    if( m_apiPicBuf[comp] )
    {
      xFree( m_apiPicBuf[comp] );
      m_apiPicBuf[comp] = NULL;
    }
  }

  for(UInt chan=0; chan<MAX_NUM_CHANNEL_TYPE; chan++)
  {
    if (m_ctuOffsetInBuffer[chan])
    {
      delete[] m_ctuOffsetInBuffer[chan];
      m_ctuOffsetInBuffer[chan] = NULL;
    }
    if (m_subCuOffsetInBuffer[chan])
    {
      delete[] m_subCuOffsetInBuffer[chan];
      m_subCuOffsetInBuffer[chan] = NULL;
    }
  }
}



Void  TComPicYuv::copyToPic (TComPicYuv*  pcPicYuvDst) /*const*/
{
  assert( m_chromaFormatIDC == pcPicYuvDst->getChromaFormat() );

  for(Int comp=0; comp<getNumberValidComponents(); comp++)
  {
    const ComponentID compId=ComponentID(comp);
    const Int width     = getWidth(compId);
    const Int height    = getHeight(compId);
    const Int strideSrc = getStride(compId);
    assert(pcPicYuvDst->getWidth(compId) == width);
    assert(pcPicYuvDst->getHeight(compId) == height);
    if (strideSrc==pcPicYuvDst->getStride(compId))
    {
      ::memcpy ( pcPicYuvDst->getBuf(compId), getBuf(compId), sizeof(Pel)*strideSrc*getTotalHeight(compId));
      pcPicYuvDst->bEdited = true;
    }
    else
    {
      const Pel *pSrc       = getAddr(compId);
            Pel *pDest      = pcPicYuvDst->getAddr(compId);
      const UInt strideDest = pcPicYuvDst->getStride(compId);

      for(Int y=0; y<height; y++, pSrc+=strideSrc, pDest+=strideDest)
      {
        ::memcpy(pDest, pSrc, width*sizeof(Pel));
      }
    }
  }

    /*assert(m_picWidth == pcPicYuvDst->getWidth(COMPONENT_Y));
    assert(m_picHeight == pcPicYuvDst->getHeight(COMPONENT_Y));
    assert(m_chromaFormatIDC == pcPicYuvDst->getChromaFormat());

    for (Int chan = 0; chan < getNumberValidComponents(); chan++)
    {
        const ComponentID ch = ComponentID(chan);
        ::memcpy(pcPicYuvDst->getBuf(ch), m_apiPicBuf[ch], sizeof(Pel) * getStride(ch) * getTotalHeight(ch));
    }*/
}


Void TComPicYuv::extendPicBorder ()
{
  if ( m_bIsBorderExtended )
  {
    return;
  }

  for(Int comp=0; comp<getNumberValidComponents(); comp++)
  {
    const ComponentID compId=ComponentID(comp);
    Pel *piTxt=getAddr(compId); // piTxt = point to (0,0) of image within bigger picture.
    const Int stride=getStride(compId);
    const Int width=getWidth(compId);
    const Int height=getHeight(compId);
    const Int marginX=getMarginX(compId);
    const Int marginY=getMarginY(compId);

    Pel*  pi = piTxt;
    // do left and right margins
    for (Int y = 0; y < height; y++)
    {
      for (Int x = 0; x < marginX; x++ )
      {
        pi[ -marginX + x ] = pi[0];
        pi[    width + x ] = pi[width-1];
      }
      pi += stride;
    }

    // pi is now the (0,height) (bottom left of image within bigger picture
    pi -= (stride + marginX);
    // pi is now the (-marginX, height-1)
    for (Int y = 0; y < marginY; y++ )
    {
      ::memcpy( pi + (y+1)*stride, pi, sizeof(Pel)*(width + (marginX<<1)) );
    }

    // pi is still (-marginX, height-1)
    pi -= ((height-1) * stride);
    // pi is now (-marginX, 0)
    for (Int y = 0; y < marginY; y++ )
    {
      ::memcpy( pi - (y+1)*stride, pi, sizeof(Pel)*(width + (marginX<<1)) );
    }
  }

  m_bIsBorderExtended = true;
}



// NOTE: This function is never called, but may be useful for developers.
Void TComPicYuv::dump (const std::string &fileName, const BitDepths &bitDepths, const Bool bAppend, const Bool bForceTo8Bit) const
{
  FILE *pFile = fopen (fileName.c_str(), bAppend?"ab":"wb");

  Bool is16bit=false;
  for(Int comp = 0; comp < getNumberValidComponents() && !bForceTo8Bit; comp++)
  {
    if (bitDepths.recon[toChannelType(ComponentID(comp))]>8)
    {
      is16bit=true;
    }
  }

  for(Int comp = 0; comp < getNumberValidComponents(); comp++)
  {
    const ComponentID  compId = ComponentID(comp);
    const Pel         *pi     = getAddr(compId);
    const Int          stride = getStride(compId);
    const Int          height = getHeight(compId);
    const Int          width  = getWidth(compId);

    if (is16bit)
    {
      for (Int y = 0; y < height; y++ )
      {
        for (Int x = 0; x < width; x++ )
        {
          UChar uc = (UChar)((pi[x]>>0) & 0xff);
          fwrite( &uc, sizeof(UChar), 1, pFile );
          uc = (UChar)((pi[x]>>8) & 0xff);
          fwrite( &uc, sizeof(UChar), 1, pFile );
        }
        pi += stride;
      }
    }
    else
    {
      const Int shift  = bitDepths.recon[toChannelType(compId)] - 8;
      const Int offset = (shift>0)?(1<<(shift-1)):0;
      for (Int y = 0; y < height; y++ )
      {
        for (Int x = 0; x < width; x++ )
        {
          UChar uc = (UChar)Clip3<Pel>(0, 255, (pi[x]+offset)>>shift);
          fwrite( &uc, sizeof(UChar), 1, pFile );
        }
        pi += stride;
      }
    }
  }

  fclose(pFile);
}

//PyTorchHelper* modelH = new PyTorchHelper("D:/OneDrive - 선문대학교/Sources/HM-16.15/working/gvcnn_half.pt", false);
//PyTorchHelper* modelQ = new PyTorchHelper("D:/OneDrive - 선문대학교/Sources/HM-16.15/working/gvcnn_quarter.pt", false);

//std::vector<cv::Mat> TComPicYuv::PredictFractionalBlocks(const cv::Mat& img, torch::jit::script::Module model)
//{
//    int img_size = img.rows * img.cols;
//
//    //cv::imwrite("test.png", img);
//
//    // 입력 벡터
//    cv::Mat float_img;
//    img.convertTo(float_img, CV_32FC1);
//
//    torch::Tensor img_tensor = torch::from_blob(float_img.data, { 1, img.rows, img.cols, 1 }, torch::kFloat).to(device);
//    img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
//    std::vector<torch::jit::IValue> input_tensors;
//    input_tensors.push_back(img_tensor);
//
//    // 예측
//    torch::Tensor netOutput = model.forward(input_tensors).toTensor();
//    netOutput = netOutput.permute({ 0, 2, 3, 1 });
//    // std::cout << netOutput.sizes() << std::endl;
//
//    // 재배치
//    auto output_tensor = netOutput.cpu().detach();
//
//    // 결과값 크기
//    size_t output_size = output_tensor.numel();
//
//    /*float* dOutput = (float*)malloc(sizeof(float) * output_size);
//    std::memcpy(dOutput, output_tensor.data_ptr(), sizeof(float) * output_size);*/
//
//    /*int _sz[] = { img.rows, img.cols, 15 };
//    cv::Mat result_img(3, _sz, CV_32FC1);
//    std::memcpy((void*)result_img.data, output_tensor.data_ptr(), sizeof(float) * output_size);*/
//
//    std::vector<cv::Mat> preds;
//    //cv::split(result_img, preds);
//    int num_img = netOutput.sizes()[3];
//    for (int i = 0; i < num_img; i++) {
//        cv::Mat result_img(img.rows, img.cols, CV_32FC1);
//        std::memcpy((void*)result_img.data, (float*)output_tensor.data_ptr() + (img_size * i), sizeof(float) * img_size);
//        //cv::Mat uchar_img;
//        //result_img.convertTo(uchar_img, CV_16SC1);
//        preds.push_back(result_img);
//
//        //cv::imwrite(format_string("%s%d%s", "test", i, ".png"), result_img);
//    }
//
//    return preds;
//}

Pel* TComPicYuv::getPicFilteredByCNN(const Int ctuRSAddr, const Int uiAbsZorderIdx/*, PyTorchHelper& model_h, PyTorchHelper& model_q*/)
{
    ComponentID ch  = COMPONENT_Y;          // luma only
    Int height      = getTotalHeight(ch);
    Int width       = getStride(ch);

    if (m_PicFilteredByCNN == NULL || bEdited)
    {
        if (m_PicFilteredByCNN == NULL)
        {
            // 버퍼가 없으면 할당한다
            m_PicFilteredByCNN = (Pel* )xMalloc(Pel, getStride(ch) * 4 * getTotalHeight(ch) * 4);
        }
        
        Pel* srcPtr = m_apiPicBuf[ch];
        Pel* dstPtr = m_PicFilteredByCNN;
        cv::Mat_<uchar> srcImg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                srcImg(i, j) = srcPtr[j];
            }
            srcPtr += width;
        }

        cv::Mat_<Short> resImg = cv::Mat(getTotalHeight(ch) * 4, getStride(ch) * 4, CV_16SC1);

        // 하프-펠
        clock_t start_t = clock();
        std::unique_ptr<PyTorchHelper> model_h = std::make_unique<PyTorchHelper>("gvcnn_half.pt", b_UseGPU);
        std::vector<cv::Mat> preds = model_h->PredictFractionalBlocks(srcImg);
        clock_t end_t = clock();

        // printf("Time for half-pel: %f\n", static_cast<double>(end_t - start_t) / CLOCKS_PER_SEC);

        cv::Mat_<Short> temp1 = preds[0] * 255 * 64;
        cv::Mat_<Short> temp2 = preds[1] * 255 * 64;
        cv::Mat_<Short> temp3 = preds[2] * 255 * 64;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resImg(i * 4,       j * 4)      = srcImg(i, j) * 64;
                resImg(i * 4 + 2,   j * 4)      = temp1(i, j);
                resImg(i * 4,       j * 4 + 2)  = temp2(i, j);
                resImg(i * 4 + 2,   j * 4 + 2)  = temp3(i, j);
            }
        }

        temp1.release();
        temp2.release();
        temp3.release();

        // 쿼터 펠
        start_t = clock();
        std::unique_ptr<PyTorchHelper> model_q = std::make_unique<PyTorchHelper>("gvcnn_quarter.pt", b_UseGPU);
        preds.clear();
        preds = model_q->PredictFractionalBlocks(srcImg);
        end_t = clock();

        // printf("Time for quarter-pel: %f\n", static_cast<double>(end_t - start_t) / CLOCKS_PER_SEC);

        cv::Mat_<Short> temp01 = preds[ 0] * 255 * 64;
        //cv::Mat_<Short> temp02 = preds[ 1] * 255 * 64;  // half
        cv::Mat_<Short> temp03 = preds[ 2] * 255 * 64;
        cv::Mat_<Short> temp04 = preds[ 3] * 255 * 64;
        cv::Mat_<Short> temp05 = preds[ 4] * 255 * 64;
        cv::Mat_<Short> temp06 = preds[ 5] * 255 * 64;
        cv::Mat_<Short> temp07 = preds[ 6] * 255 * 64;
        //cv::Mat_<Short> temp08 = preds[ 7] * 255 * 64;  // half
        cv::Mat_<Short> temp09 = preds[ 8] * 255 * 64;
        //cv::Mat_<Short> temp10 = preds[ 9] * 255 * 64;  // half
        cv::Mat_<Short> temp11 = preds[10] * 255 * 64;
        cv::Mat_<Short> temp12 = preds[11] * 255 * 64;
        cv::Mat_<Short> temp13 = preds[12] * 255 * 64;
        cv::Mat_<Short> temp14 = preds[13] * 255 * 64;
        cv::Mat_<Short> temp15 = preds[14] * 255 * 64;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resImg(i * 4,       j * 4 + 1)      = temp01(i, j);
                //resImg(i * 4,       j * 4 + 2)      = temp02(i, j); // half
                resImg(i * 4,       j * 4 + 3)      = temp03(i, j);
                resImg(i * 4 + 1,   j * 4)          = temp04(i, j);
                resImg(i * 4 + 1,   j * 4 + 1)      = temp05(i, j);
                resImg(i * 4 + 1,   j * 4 + 2)      = temp06(i, j);
                resImg(i * 4 + 1,   j * 4 + 3)      = temp07(i, j);
                //resImg(i * 4 + 2,   j * 4)          = temp08(i, j); // half
                resImg(i * 4 + 2,   j * 4 + 1)      = temp09(i, j);
                //resImg(i * 4 + 2,   j * 4 + 2)      = temp10(i, j); // half
                resImg(i * 4 + 2,   j * 4 + 3)      = temp11(i, j);
                resImg(i * 4 + 3,   j * 4)          = temp12(i, j);
                resImg(i * 4 + 3,   j * 4 + 1)      = temp13(i, j);
                resImg(i * 4 + 3,   j * 4 + 2)      = temp14(i, j);
                resImg(i * 4 + 3,   j * 4 + 3)      = temp15(i, j);
            }
        }

        temp01.release();
        //temp02.release();
        temp03.release();
        temp04.release();
        temp05.release();
        temp06.release();
        temp07.release();
        //temp08.release();
        temp09.release();
        //temp10.release();
        temp11.release();
        temp12.release();
        temp13.release();
        temp14.release();
        temp15.release();
        preds.clear();

        // cudaDeviceReset();

        // 클리핑
        for (int i = 0; i < resImg.rows; i++) {
            for (int j = 0; j < resImg.cols; j++) {
                if (resImg(i, j) >= 255 * 64) {
                    resImg(i, j) = 255 * 64;
                }
                if (resImg(i, j) < 0) {
                    resImg(i, j) = 0;
                }
            }
        }

        for (int i = 0; i < height * 4; i++) {
            for (int j = 0; j < width * 4; j++) {
                dstPtr[j] = resImg(i, j);
            }
            dstPtr += width * 4;
        }
        bEdited = false;
    }

    int totalOffset = m_ctuOffsetInBuffer[ch == 0 ? 0 : 1][ctuRSAddr]
        + m_subCuOffsetInBuffer[ch == 0 ? 0 : 1][g_auiZscanToRaster[uiAbsZorderIdx]];
    
    int rows = totalOffset / width;
    int cols = totalOffset % width;

    return m_PicFilteredByCNN + m_marginY * 4 * width * 4 + m_marginX * 4 + rows * 4 * width * 4 + cols * 4;
    //return Pel();
}

Pel* TComPicYuv::getPicFilteredByCNNv2(const Int ctuRSAddr, const Int uiAbsZorderIdx)
{
    ComponentID ch = COMPONENT_Y;          // luma only
    Int height = getTotalHeight(ch);
    Int width = getStride(ch);

    if (m_PicFilteredByCNN == NULL || bEdited)
    {
        if (m_PicFilteredByCNN == NULL)
        {
            // 버퍼가 없으면 할당한다
            m_PicFilteredByCNN = (Pel*)xMalloc(Pel, getStride(ch) * 4 * getTotalHeight(ch) * 4);
        }

        Pel* srcPtr = m_apiPicBuf[ch];
        Pel* dstPtr = m_PicFilteredByCNN;
        cv::Mat_<uchar> srcImg = cv::Mat(getTotalHeight(ch), getStride(ch), CV_8UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                srcImg(i, j) = srcPtr[j];
            }
            srcPtr += width;
        }

        cv::Mat_<Short> resImg = cv::Mat(getTotalHeight(ch) * 4, getStride(ch) * 4, CV_16SC1);

        // 하프-펠
        clock_t start_t = clock();
        std::unique_ptr<PyTorchHelper> fbeNet = std::make_unique<PyTorchHelper>("traced_fractional_enhancement_model.pt", b_UseGPU);
        fbeNet->PredictFractionalBlocks(srcImg, dstPtr);
        clock_t end_t = clock();

        // 클리핑
        size_t dst_size = getStride(ch) * 4 * getTotalHeight(ch) * 4;
        for (int i = 0; i < dst_size; i++) {
            if (dstPtr[i] >= 255 * 64) {
                dstPtr[i] = 255 * 64;
            }
            if (dstPtr[i] < 0) {
                dstPtr[i] = 0;
            }
        }
        bEdited = false;
    }

    int totalOffset = m_ctuOffsetInBuffer[ch == 0 ? 0 : 1][ctuRSAddr]
        + m_subCuOffsetInBuffer[ch == 0 ? 0 : 1][g_auiZscanToRaster[uiAbsZorderIdx]];

    int rows = totalOffset / width;
    int cols = totalOffset % width;

    return m_PicFilteredByCNN + m_marginY * 4 * width * 4 + m_marginX * 4 + rows * 4 * width * 4 + cols * 4;
}

//! \}
