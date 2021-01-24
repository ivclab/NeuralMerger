/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */
// Modified:   Yi-Ming Chan, 2018

#ifndef INCLUDE_CAFFEPARA_H_
#define INCLUDE_CAFFEPARA_H_

#include <string>
#include <vector>

#include "../include/Common.h"
#include "../include/Matrix.h"

// define ENUM class for file formats
// description:
//   assume <x> is the minimal number of bits to store an index.
//   FileFormat::Raw: each index is stored in <ceil(x / 8)> bytes
//   FileFormat::Compact: each index is stored in <x / 8> bytes
enum class ENUM_AsmtEnc
{
  Raw,
  Compact
};

// define ENUM class for layer types
enum class ENUM_LyrType
{
  Conv,
  Pool,
  AvgPool,
  FCnt,
  ReLU,
  LoRN,
  Drpt,
  SMax,
  PReLU,
  Shortcut
};

// define the arrangement of the weight
// Tensorflow use height, width, input channel, output channel
// Caffe use output channel, input channel, height, width
enum class ENUM_LyrArrangement
{
  ChannelHeightWidth,
  HeightWidthChannel
};

// define structure <LayerInfo> and <LayerInfoLst>
typedef struct
{
  ENUM_LyrType type; // layer type
  int padSiz;        // number of padded pixels (on each side)
  int knlSiz;        // width/height of the convolutional kernel
  int knlCnt;        // number of convolutional kernels
  int grpCnt;        // number of source feature map groups
  int stride;        // convolution stride / spatial step
  int nodCnt;        // number of target neurons (for the <fcnt> layer)
  int lrnSiz;        // local response patch size
  float lrnAlp;      // local response normalization - alpha
  float lrnBet;      // local response normalization - beta
  float lrnIni;      // local response normalization - initial value
  bool lrnAcr;       // local response normalization - across channel
  float drpRat;      // dropout ratio (how many neurons are preserved)
  int shortcutFrom;  // index of layer of shortcut from
  ENUM_LyrArrangement arrang;
} LayerInfo;
typedef std::vector<LayerInfo> LayerInfoLst;

// define structure <LayerPara> and <LayerParaLst>
typedef struct
{
  Matrix<float> convKnlLst;
  Matrix<float> fcntWeiMat;
  Matrix<float> biasVec;
  Matrix<float> ctrdLst;
  Matrix<float> alphaVec; // For PReLU vector
  Matrix<uint8_t> asmtLst;
} LayerPara;
typedef std::vector<LayerPara> LayerParaLst;

class CaffePara
{
public:
  // Default constructor
  CaffePara(void) : useMatlab(true) {}
  // initialize parameters for quantization
  void Init(const std::string &dirPathSrc, const std::string &filePfxSrc);
  // configure all layers according to the <AlexNet> settings
  void ConfigLayer_AlexNet(void);
  // configure all layers according to the <CaffeNet> settings
  void ConfigLayer_CaffeNet(void);
  // configure all layers according to the <VggCnnS> settings
  void ConfigLayer_VggCnnS(void);
  // configure all layers according to the <VGG16> settings
  void ConfigLayer_VGG16(void);
  // configure all layers according to the <VGG16 AveragePooling> settings
  void ConfigLayer_VGG16Avg(void);
  // configure all layers according to the <CaffeNetFGB> settings
  void ConfigLayer_CaffeNetFGB(void);
  // configure all layers according to the <CaffeNetFGD> settings
  void ConfigLayer_SoundCNN(void);
  // configure SoundCNN layers
  // https://medium.com/@awjuliani/recognizing-sounds-a-deep-learning-case-study-1bc37444d44d
  void ConfigLayer_LeNet(void);
  // configure LeNet layers
  void ConfigLayer_ZFNet(void);
  // configure ZFNet
  void ConfigLayer_CaffeNetFGD(void);
  // configure SphereFace20
  void ConfigLayer_SphereFace20(void);
  // load layer parameters from files
  bool LoadLayerPara(const bool enblAprx, const ENUM_AsmtEnc asmtEnc);
  // load layer parameters from list file
  bool LoadLayerPara(const bool enblAprx, const ENUM_AsmtEnc asmtEnc, const std::string modelFile);
  // load both quantized and normal parameters from files for debug. Added by Yi-Ming Chan
  bool LoadLayerPara(const ENUM_AsmtEnc asmtEnc);
  // bool variable for 0 or 1 index parameters in assigment vector
  bool useMatlab;
  // convert raw-encoded index files to compact-encoded
  bool CvtAsmtEnc(const ENUM_AsmtEnc asmtEncSrc, const ENUM_AsmtEnc asmtEncDst);
  // convert raw-encoded index files to compact-encoded with model list
  bool CvtAsmtEnc(const ENUM_AsmtEnc asmtEncSrc, const ENUM_AsmtEnc asmtEncDst, const std::vector<std::string> modelFile);

public:
  // main directory for data import/export
  std::string dirPath;
  // file name prefix
  std::string filePfx;
  // number of layers
  int layerCnt;
  // number of input feature map channels
  int imgChnIn;
  // input feature map height
  int imgHeiIn;
  // input feature map width
  int imgWidIn;
  // all layers' basic information
  LayerInfoLst layerInfoLst;
  // all layers' parameters
  LayerParaLst layerParaLst;

private:
  // determine the proper value for <bitCntPerEle>
  int CalcBitCntPerEle(const Matrix<uint8_t> &asmtLst);
  // configure each layer type
  void ConfigConvLayer(LayerInfo *pLayerInfo, const int padSiz,
                       const int knlSiz, const int knlCnt, const int grpCnt, const int stride);
  void ConfigPoolLayer(LayerInfo *pLayerInfo,
                       const int padSiz, const int knlSiz, const int stride);
  void ConfigAvgPoolLayer(LayerInfo *pLayerInfo,
                          const int padSiz, const int knlSiz, const int stride);
  void ConfigFCntLayer(LayerInfo *pLayerInfo, const int nodCnt);
  void ConfigReLuLayer(LayerInfo *pLayerInfo);
  void ConfigLoRNLayer(LayerInfo *pLayerInfo, const int lrnSiz,
                       const float lrnAlp, const float lrnBet, const float lrnIni,
                       const bool acrosschannel = true);
  void ConfigDrptLayer(LayerInfo *pLayerInfo, const float drpRat);
  void ConfigSMaxLayer(LayerInfo *pLayerInfo);
  void ConfigPReLULayer(LayerInfo *pLayerInfo);
  void ConfigShortcutLayer(LayerInfo *pLayerInfo, const int idxFrom); // Add shortcut layer
};

#endif // INCLUDE_CAFFEPARA_H_
