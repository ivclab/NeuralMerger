/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/CaffeEva.h"

#include "../include/BlasWrapper.h"
#include "../include/CaffePara.h"
#include "../include/Common.h"
#include "../include/FileIO.h"

#ifdef ENBL_ANDROID_DBGR
#include "share.h"
#define SHR_PRINTF LOGI
#else
#define SHR_PRINTF printf
#endif // ENDIF: ENBL_ANDROID_DBG

// initialize constant variables
const int kDataCntInBatch = 8; // number of images in each batch
const int kBatchCntProc = 120;  // number of batches
const int kLablCntPerData = 1;  // number of predicted labels per image

CaffeEva::~CaffeEva(void)
{
  // release dynamically allocated memory
  delete[] featMapLst;
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    FeatBufStrLst &featBufStrLst = featBufStrMat[layerInd];
    for (std::size_t bufInd = 0; bufInd < featBufStrLst.size(); bufInd++)
    {
      // skip if no memory is allocated to the current buffer
      if (featBufStrLst[bufInd].pFeatBuf != nullptr)
      {
        delete featBufStrLst[bufInd].pFeatBuf;
        featBufStrLst[bufInd].pFeatBuf = nullptr;
      } // ENDIF: featBufStrLst
    }   // ENDFOR: bufInd
  }     // ENDFOR: layerInd

  // destory objects: <caffeParaObj>
}

void CaffeEva::Init(const bool enblAprxSrc)
{
  // initialize <enblAprx>
  enblAprx = enblAprxSrc;

  // initialize all stop-watches
  swAllLayers.Reset();
  swConvLayer.Reset();
  swPoolLayer.Reset();
  swAvgPoolLayer.Reset();
  swFCntLayer.Reset();
  swReLuLayer.Reset();
  swLoRNLayer.Reset();
  swDrptLayer.Reset();
  swSMaxLayer.Reset();
  swShortcutLayer.Reset();
  swPReLULayer.Reset();
  swCompLkupTblConv.Reset();
  swEstiInPdValConv.Reset();
  swCompLkupTblFCnt.Reset();
  swEstiInPdValFCnt.Reset();
  swDebugTimePri.Reset();
  swDebugTimeSec.Reset();
}

void CaffeEva::SetModelName(const std::string &modelNameSrc)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::SetDirPath()\n");

  // specify the model name
  modelName = modelNameSrc;
}

void CaffeEva::SetModelPath(
    const std::string &dirPathMainSrc, const std::string &fileNamePfxSrc)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::SetDirPath()\n");

  // specify the main directory path and file name prefix
  dirPathMain = dirPathMainSrc;
  fileNamePfx = fileNamePfxSrc;
}

bool CaffeEva::LoadOutputonly(const std::string& fileName)
{
   // declare auxiliary variables
  bool succFlg;

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadOutputonly()\n");

  // load samples in the evaluation subset
  succFlg = FileIO::ReadBinFile(fileName.c_str(), &outputMat);
  if (!succFlg)
  { // failed
    return false;
  } // ENDIF: succFlg
  
  return true;
}

void CaffeEva::EvaluateOutput(int iLayer)
{

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::EvaluateOutput()\n");
		
  // The first batch of the output
  int iNumberOutput = kDataCntInBatch;

  double differences[kDataCntInBatch] = {0};

  // Assume the last layer is the output layer
  Matrix<float> *Output = GetFeatMap(iLayer);
  // Convert to from (Out, H, W, In) to (Out, In, H, W)
  if( Output->GetDimCnt() == 4 )
     Output->Permute(0,3,1,2);
  
  Matrix<float> *OutputGroundTruth = &outputMat;
  printf("Differences =");
  for(int i=0; i < kDataCntInBatch; i++)
  {
	  differences[i]=0;
	  int iOutElements = Output->GetEleCnt();
	  int iElePerImg = iOutElements / kDataCntInBatch;
    float *fpOutput = Output->GetDataPtr();
    float *fpGround = OutputGroundTruth->GetDataPtr();
	  for(int j=0; j < iElePerImg; j++)
	  {
		  double diffs = fabs(fpOutput[i * iElePerImg + j] - fpOutput[i * iElePerImg + j]);
      double ratio = fabs(diffs/fpOutput[i * iElePerImg + j]);
      differences[i] += diffs;
      
	  }

	  printf(" %f", differences[i]);
  }

  printf("\n");
}

// Leave the label unknown. Good for debug
bool CaffeEva::LoadDatasetonly(const std::string& dirPathData)
{
 // declare auxiliary variables
  const int kStrBufLen = 256;
  char strBuf[kStrBufLen];
  bool succFlg;

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadDatasetonly()\n");

  // load samples in the evaluation subset
  snprintf(strBuf, kStrBufLen, "%s/dataMatTst.single.bin", dirPathData.c_str());
  succFlg = FileIO::ReadBinFile(strBuf, &dataLst);
  if (!succFlg)
  { // failed
    return false;
  } // ENDIF: succFlg
  
  return true;
}

bool CaffeEva::LoadDataset(const std::string &dirPathData)
{
  // declare auxiliary variables
  const int kStrBufLen = 256;
  char strBuf[kStrBufLen];
  bool succFlg;

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadDataset()\n");

  // load samples in the evaluation subset
  snprintf(strBuf, kStrBufLen, "%s/dataMatTst.single.bin", dirPathData.c_str());
  succFlg = FileIO::ReadBinFile(strBuf, &dataLst);
  if (!succFlg)
  { // failed
    return false;
  } // ENDIF: succFlg

  // load samples' ground-truth labels in the evaluation subset
  snprintf(strBuf, kStrBufLen, "%s/lablVecTst.uint16.bin", dirPathData.c_str());
  succFlg = FileIO::ReadBinFile(strBuf, &lablVecGrth);
  if (!succFlg)
  { // failed
    return false;
  } // ENDIF: succFlg

  return true;
}

bool CaffeEva::LoadCaffePara(void)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::LoadCaffePara()\n");

  // initialize <caffeParaObj>
  caffeParaObj.Init(dirPathMain, fileNamePfx);

  // load each layer's basic information
  if (modelName == "AlexNet")
  {
    caffeParaObj.ConfigLayer_AlexNet();
  }
  else if (modelName == "CaffeNet")
  {
    caffeParaObj.ConfigLayer_CaffeNet();
  }
  else if (modelName == "VggCnnS")
  {
    caffeParaObj.ConfigLayer_VggCnnS();
  }
  else if (modelName == "VGG16")
  {
    caffeParaObj.ConfigLayer_VGG16();
  }
  else if (modelName == "CaffeNetFGB")
  {
    caffeParaObj.ConfigLayer_CaffeNetFGB();
  }
  else if (modelName == "CaffeNetFGD")
  {
    caffeParaObj.ConfigLayer_CaffeNetFGD();
  }
  else if (modelName == "VGG16Avg")
  {
    caffeParaObj.ConfigLayer_VGG16Avg();
  }
  else if (modelName == "SoundCNN")
  {
    caffeParaObj.ConfigLayer_SoundCNN();
  }
  else if (modelName == "LeNet")
  {
    caffeParaObj.ConfigLayer_LeNet();
  }
  else if (modelName == "ZFNet")
  {
    caffeParaObj.ConfigLayer_ZFNet();
  }
  else if (modelName == "SphereFace20")
  {
    caffeParaObj.ConfigLayer_SphereFace20();
  }
  else
  {
    printf("[ERROR] unrecognized caffe model name: %s\n", modelName.c_str());
    return false;
  } // ENDIF: modelName

  // load each layer's detailed parameters
  bool succFlg = caffeParaObj.LoadLayerPara(enblAprx, ENUM_AsmtEnc::Compact);
  if (!succFlg)
  { // failed
    return false;
  } // ENDIF: succFlg

  // prepare feature map and buffers for each layer
  PrepFeatMap();
  PrepFeatBuf();
  if (enblAprx)
  {
    PrepCtrdBuf();
    PrepAsmtBuf();
  } // ENDIF: enblAprx

  return true;
}

void CaffeEva::ExecForwardPass(void)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::ExecForwardPass()\n");

  // initialize stop-watches for each layer
  swIndvLayerLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    swIndvLayerLst[layerInd].Reset();
  } // ENDFOR: layerInd

  // pack samples into batches and then execute the forward pass
  int dataCnt = dataLst.GetDimLen(0);
  int dataIndL;
  int dataIndU;
  int batchCnt = (dataCnt + kDataCntInBatch - 1) / kDataCntInBatch;
  lablVecPred.Create(dataCnt, kLablCntPerData, 1, 1); 
  int ibatch = std::min(kBatchCntProc, batchCnt);
  for (int batchInd = 0; batchInd < ibatch; batchInd++)
  {
    printf("processing the %d-th batch\n", batchInd + 1);

    // check whether is the last batch
    if (batchInd < batchCnt - 1)
    {
      dataIndL = kDataCntInBatch * batchInd;
      dataIndU = dataIndL + (kDataCntInBatch - 1);
    }
    else
    {
      dataIndU = dataCnt - 1;
      dataIndL = dataIndU - (kDataCntInBatch - 1);
    } // ENDIF: batchInd

    // convert samples' feature vectors into the input feature map
    CvtDataLstToFeatMap(dataIndL, dataIndU, dataLst, &(featMapLst[0]));

    // execute the forward pass
    bool isFirstFCnt = true;
    for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
    {
      // permute dimensions for the first fully-connected layer
      const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
      if (layerInfo.arrang == ENUM_LyrArrangement::HeightWidthChannel && enblAprx)
      {
        isFirstFCnt = false; // No need to permute anymore
      }
      if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt))
      {
        featMapLst[layerInd].Permute(0, 3, 1, 2);
      } // ENDIF: isFirstFCnt

      // compute the target layer's activation 
      swIndvLayerLst[layerInd].Resume();
      if(layerInfo.type != ENUM_LyrType::Shortcut)
      {
        CalcFeatMap(featMapLst[layerInd], layerInd, &(featMapLst[layerInd + 1]));
      }
      else
      {
        swShortcutLayer.Resume();
        CalcFeatMap_Shortcut(featMapLst[layerInd], featMapLst[layerInfo.shortcutFrom+1], 
                             &(featMapLst[layerInd + 1]));
        swShortcutLayer.Pause();
      }
      swIndvLayerLst[layerInd].Pause();

      // permute dimensions for the first fully-connected layer
      if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt))
      {
        isFirstFCnt = false;
        int m = featMapLst[layerInd].GetDimLen(0);
        int n = featMapLst[layerInd].GetDimLen(1);
        int p = featMapLst[layerInd].GetDimLen(2);
        int q = featMapLst[layerInd].GetDimLen(3);
        featMapLst[layerInd].Resize(m, p, q, n);
      } // ENDIF: isFirstFCnt
    }   // ENDIF: layerInd

    // convert the output feature map into samples' predicted labels
    CvtFeatMapToLablVec(dataIndL,
                        dataIndU, featMapLst[caffeParaObj.layerCnt], &lablVecPred);

  } // ENDFOR: batchInd
}

void CaffeEva::ExecForwardPass(
    const Matrix<float> &imgDataIn, Matrix<float> *pProbVecOut)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::ExecForwardPass()\n");

  // initialize stop-watches for each layer
  swIndvLayerLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    swIndvLayerLst[layerInd].Reset();
  } // ENDFOR: layerInd

  // copy <dataLstIn> to the input feature map
  featMapLst[0].Permute(0, 3, 1, 2);
  memcpy(featMapLst[0].GetDataPtr(),
         imgDataIn.GetDataPtr(), sizeof(float) * imgDataIn.GetEleCnt());
  featMapLst[0].Permute(0, 2, 3, 1);

  // execute the forward pass
  bool isFirstFCnt = true;
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    printf("layerInd = %d\n", layerInd);

    // permute dimensions for the first fully-connected layer
    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
    if (layerInfo.arrang == ENUM_LyrArrangement::HeightWidthChannel && enblAprx)
    {
      isFirstFCnt = false; // No need to permute anymore
    }
    if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt))
    {
      featMapLst[layerInd].Permute(0, 3, 1, 2);
    } // ENDIF: isFirstFCnt

    // compute the target layer's activation
    swIndvLayerLst[layerInd].Resume();
    CalcFeatMap(featMapLst[layerInd], layerInd, &(featMapLst[layerInd + 1]));
    swIndvLayerLst[layerInd].Pause();

    // permute dimensions for the first fully-connected layer
    if (isFirstFCnt && (layerInfo.type == ENUM_LyrType::FCnt))
    {
      isFirstFCnt = false;
      int m = featMapLst[layerInd].GetDimLen(0);
      int n = featMapLst[layerInd].GetDimLen(1);
      int p = featMapLst[layerInd].GetDimLen(2);
      int q = featMapLst[layerInd].GetDimLen(3);
      featMapLst[layerInd].Resize(m, p, q, n);
    } // ENDIF: isFirstFCnt
  }   // ENDIF: layerInd

  // extract <dataLstOut> from the output feature map
  pProbVecOut->Resize(featMapLst[caffeParaObj.layerCnt].GetEleCnt());
  memcpy(pProbVecOut->GetDataPtr(),
         featMapLst[caffeParaObj.layerCnt].GetDataPtr(),
         sizeof(float) * pProbVecOut->GetEleCnt());
}

void CaffeEva::CalcPredAccu(void)
{
  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::CalcPredAccu()\n");

  // initialize counters for accuracy computation
  Matrix<uint32_t> accuCntLst(kLablCntPerData);
  Matrix<float> accuScrLst(kLablCntPerData);
  memset(accuCntLst.GetDataPtr(), 0, sizeof(uint32_t) * accuCntLst.GetEleCnt());
  memset(accuScrLst.GetDataPtr(), 0, sizeof(float) * accuScrLst.GetEleCnt());

  // compute the total number of correctly predicted class labels
  int dataCnt = kDataCntInBatch * kBatchCntProc;
  int iMaxCnt = dataLst.GetDimLen(0);
  if (dataCnt > iMaxCnt)
    dataCnt = iMaxCnt;
  const uint16_t *lablPtrGrth = lablVecGrth.GetDataPtr();
  const uint16_t *lablPtrPred = lablVecPred.GetDataPtr();
  uint32_t *accuCntVec = accuCntLst.GetDataPtr();
  float *accuScrVec = accuScrLst.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++)
    {
      uint16_t lablVal_Pred = lablPtrPred[dataInd * kLablCntPerData + lablInd];
      if (lablPtrGrth[dataInd] == lablVal_Pred)
      {
        accuCntVec[lablInd]++;
      } // ENDIF: lablPtrGrth
    }   // ENDFOR: lablInd
  }     // ENDFOR: dataInd
  for (int lablInd = 1; lablInd < kLablCntPerData; lablInd++)
  {
    accuCntVec[lablInd] += accuCntVec[lablInd - 1];
  } // ENDFOR: lablInd
  for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++)
  {
    accuScrVec[lablInd] = static_cast<double>(accuCntVec[lablInd]) / dataCnt;
    printf("ACCURACY@%d: %d, %.2f%%\n",
           lablInd + 1, accuCntVec[lablInd], accuScrVec[lablInd] * 100);
  } // ENDFOR: lablInd
}

float CaffeEva::DispElpsTime(void)
{
  // get total computation time
  float timeTotal = swAllLayers.GetTime();

  // display the elapsed time of each stop-watch
  SHR_PRINTF("swAllLayers: %.4f (s)\n", timeTotal);
  SHR_PRINTF("swConvLayer: %.4f (s)\n", swConvLayer.GetTime());
  SHR_PRINTF("swPoolLayer: %.4f (s)\n", swPoolLayer.GetTime());
  SHR_PRINTF("swAvgPoolLayer: %.4f (s)\n", swAvgPoolLayer.GetTime());
  SHR_PRINTF("swFCntLayer: %.4f (s)\n", swFCntLayer.GetTime());
  SHR_PRINTF("swReLuLayer: %.4f (s)\n", swReLuLayer.GetTime());
  SHR_PRINTF("swPReLULayer: %.4f (s)\n", swPReLULayer.GetTime());
  SHR_PRINTF("swShortcutLayer: %.4f (s)\n", swShortcutLayer.GetTime());
  SHR_PRINTF("swLoRNLayer: %.4f (s)\n", swLoRNLayer.GetTime());
  SHR_PRINTF("swDrptLayer: %.4f (s)\n", swDrptLayer.GetTime());
  SHR_PRINTF("swSMaxLayer: %.4f (s)\n", swSMaxLayer.GetTime());
  SHR_PRINTF("swCompLkupTblConv: %.4f (s)\n", swCompLkupTblConv.GetTime());
  SHR_PRINTF("swEstiInPdValConv: %.4f (s)\n", swEstiInPdValConv.GetTime());
  SHR_PRINTF("swCompLkupTblFCnt: %.4f (s)\n", swCompLkupTblFCnt.GetTime());
  SHR_PRINTF("swEstiInPdValFCnt: %.4f (s)\n", swEstiInPdValFCnt.GetTime());
  SHR_PRINTF("swDebugTimePri: %.4f (s)\n", swDebugTimePri.GetTime());
  SHR_PRINTF("swDebugTimeSec: %.4f (s)\n", swDebugTimeSec.GetTime());

  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    SHR_PRINTF("swIndvLayerLst #%2d: %.4f (s)\n",
               layerInd + 1, swIndvLayerLst[layerInd].GetTime());
  } // ENDFOR: layerInd

  // re-initialize each stop-watch
  Init(enblAprx);

  return timeTotal;
}

void CaffeEva::PrepFeatMap(void)
{
  // allocate memory space for <featMapSizLst>
  featMapSizLst.resize(caffeParaObj.layerCnt + 1);

  // determine the size of the input feature map
  featMapSizLst[0].dataCnt = kDataCntInBatch;
  featMapSizLst[0].imgHei = caffeParaObj.imgHeiIn;
  featMapSizLst[0].imgWid = caffeParaObj.imgWidIn;
  featMapSizLst[0].imgChn = caffeParaObj.imgChnIn;

  // determine the size of the remaining feature maps
  for (int layerInd = 1; layerInd <= caffeParaObj.layerCnt; layerInd++)
  {
    // obtain reference to previous/current feature map size
    const FeatMapSiz &featMapSizPrev = featMapSizLst[layerInd - 1];
    FeatMapSiz &featMapSizCurr = featMapSizLst[layerInd];
    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd - 1];

    // obtain basic variables
    int dataCnt = featMapSizPrev.dataCnt;
    int imgHeiPrev = featMapSizPrev.imgHei;
    int imgWidPrev = featMapSizPrev.imgWid;
    int imgChnPrev = featMapSizPrev.imgChn;
    int padSiz = layerInfo.padSiz;
    int knlSiz = layerInfo.knlSiz;
    int knlCnt = layerInfo.knlCnt;
    int stride = layerInfo.stride;
    double strideDbl = static_cast<double>(stride);
    int nodCnt = layerInfo.nodCnt;

    // compute the feature map size
    switch (layerInfo.type)
    {
    case ENUM_LyrType::Conv:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei = (imgHeiPrev + 2 * padSiz - (knlSiz - 1) - 1 + stride) / strideDbl; // Fast Ceilling operator with overflow into consideration (the last -1 and + 1)
      featMapSizCurr.imgWid = ceil((imgWidPrev + 2 * padSiz - (knlSiz - 1)) / strideDbl);
      featMapSizCurr.imgChn = knlCnt;
      break;
    case ENUM_LyrType::Pool:
    case ENUM_LyrType::AvgPool:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei =
          ceil((imgHeiPrev + 2 * padSiz - (knlSiz)) / strideDbl) + 1;
      featMapSizCurr.imgWid =
          ceil((imgWidPrev + 2 * padSiz - (knlSiz)) / strideDbl) + 1;
      featMapSizCurr.imgChn = imgChnPrev;
      break;
    case ENUM_LyrType::FCnt:
      featMapSizCurr.dataCnt = dataCnt;
      featMapSizCurr.imgHei = 1;
      featMapSizCurr.imgWid = 1;
      featMapSizCurr.imgChn = nodCnt;
      break;
    case ENUM_LyrType::ReLU:
      // fall through
    case ENUM_LyrType::PReLU:
      // fall through
    case ENUM_LyrType::Shortcut:
      // fall through
    case ENUM_LyrType::LoRN:
      // fall through
    case ENUM_LyrType::Drpt:
      // fall through
    case ENUM_LyrType::SMax:
      featMapSizCurr = featMapSizPrev;
      break;
    default:
      printf("[ERROR] invalid layer type\n");
      return;
    } // ENDSWITCH: layerInfo
  }   // ENDFOR: layerInd

  // allocate memory for each feature map
  featMapLst = new Matrix<float>[caffeParaObj.layerCnt + 1];
  for (int layerInd = 0; layerInd <= caffeParaObj.layerCnt; layerInd++)
  {
    const FeatMapSiz &featMapSiz = featMapSizLst[layerInd];
    featMapLst[layerInd].Create(featMapSiz.dataCnt,
                                featMapSiz.imgHei, featMapSiz.imgWid, featMapSiz.imgChn);
  } // ENDFOR: layerInd

  // display the feature map size
  for (int layerInd = 0; layerInd <= caffeParaObj.layerCnt; layerInd++)
  {
    const FeatMapSiz &featMapSiz = featMapSizLst[layerInd];
    const Matrix<float> &featMap = featMapLst[layerInd];
    float memUsage = featMap.GetEleCnt() * 4 / 1024.0 / 1024.0;
    printf("layer #%2d: %4d x %4d x %4d x %4d (%6.3f MB)\n",
           layerInd, featMapSiz.dataCnt, featMapSiz.imgHei,
           featMapSiz.imgWid, featMapSiz.imgChn, memUsage);
  } // ENDFOR: layerInd
}

void CaffeEva::PrepFeatBuf(void)
{
  // define a template for <FeatBufStr>
  static FeatBufStr featBufStr;

  // determine the size of each layer's feature buffer
  featBufStrMat.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    // obtain reference to previous/current feature map size
    const FeatMapSiz &featMapSizCurr = featMapSizLst[layerInd];
    const FeatMapSiz &featMapSizNext = featMapSizLst[layerInd + 1];
    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
    FeatBufStrLst &featBufStrLst = featBufStrMat[layerInd];

    // obtain basic variables
    int dataCnt = featMapSizCurr.dataCnt;
    int imgHeiCurr = featMapSizCurr.imgHei;
    int imgWidCurr = featMapSizCurr.imgWid;
    int imgChnCurr = featMapSizCurr.imgChn;
    int imgHeiNext = featMapSizNext.imgHei;
    int imgWidNext = featMapSizNext.imgWid;
    int knlSiz = layerInfo.knlSiz;
    int knlCnt = layerInfo.knlCnt;
    int grpCnt = layerInfo.grpCnt;
    int lrnSiz = layerInfo.lrnSiz;
    int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
    int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);

    // compute the feature buffer size
    featBufStrLst.clear();
    switch (layerInfo.type)
    {
    case ENUM_LyrType::Conv:
      // feature buffer #0: <featMapSrcPrm>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
                  dataCnt, imgChnCurr, imgHeiCurr, imgWidCurr);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <featMapSrcRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
                  imgChnCurr / grpCnt * knlSiz * knlSiz, imgHeiNext * imgWidNext);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #2: <featMapDstRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::PrecComp,
                  knlCnt / grpCnt, imgHeiNext * imgWidNext);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #3: <featMapSrcPerGrp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
                  dataCnt, imgHeiCurr, imgWidCurr, imgChnCurr / grpCnt);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #4: <inPdMat>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
                  dataCnt * imgHeiCurr * imgWidCurr, subSpaceCnt, ctrdCntPerSpace);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::FCnt:
      // feature buffer #0: <featMapSrcRsp>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
                  dataCnt, imgChnCurr * imgHeiCurr * imgWidCurr);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <inPdMat>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::AprxComp,
                  dataCnt, subSpaceCnt, ctrdCntPerSpace);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::LoRN:
      // feature buffer #0: <featVecSrcExt>
      InitFeatBuf(&featBufStr,
                  ENUM_BufUsage::GnrlComp, imgChnCurr + lrnSiz - 1);
      featBufStrLst.push_back(featBufStr);
      // feature buffer #1: <loclSumLst>
      InitFeatBuf(&featBufStr, ENUM_BufUsage::GnrlComp, imgChnCurr);
      featBufStrLst.push_back(featBufStr);
      break;
    case ENUM_LyrType::Pool:
      // fall through
    case ENUM_LyrType::AvgPool:
      // fall through
    case ENUM_LyrType::ReLU:
      // fall through
    case ENUM_LyrType::Drpt:
      // fall through
    case ENUM_LyrType::PReLU:
      // fall through
    case ENUM_LyrType::Shortcut:
      // fall through
    case ENUM_LyrType::SMax:
      // do nothing
      break;
    default:
      printf("[ERROR] invalid layer type\n");
      return;
    } // ENDSWITCH: layerInfo
  }   // ENDFOR: layerInd

  // display the feature buffer size
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    // obtain a constant reference to the feature buffer list
    FeatBufStrLst &featBufStrLst = featBufStrMat[layerInd];

    // display the feature buffer size
    printf("layer #%2d: \n", layerInd + 1);
    for (std::size_t bufInd = 0; bufInd < featBufStrLst.size(); bufInd++)
    {
      FeatBufStr &featBufStr = featBufStrLst[bufInd];

      // check whether current buffer will be used in the future
      if ((enblAprx && (featBufStr.usage == ENUM_BufUsage::PrecComp)) ||
          (!enblAprx && (featBufStr.usage == ENUM_BufUsage::AprxComp)))
      {
        continue;
      } // ENDIF: enblAprx

      // allocate memory space for the current buffer
      featBufStr.pFeatBuf = new Matrix<float>();
      featBufStr.pFeatBuf->Create(featBufStr.dimCnt, featBufStr.dimLenLst);

      // display the memory consumption of the current buffer
      printf("  buffer #%lu: ", bufInd + 1);
      float memUsage = featBufStr.pFeatBuf->GetEleCnt() * 4 / 1024.0 / 1024.0;
      for (int dimInd = 0; dimInd < featBufStr.dimCnt; dimInd++)
      {
        if (dimInd < featBufStr.dimCnt - 1)
        {
          printf("%4d x ", featBufStr.dimLenLst[dimInd]);
        }
        else
        {
          printf("%4d (%6.3f MB)\n", featBufStr.dimLenLst[dimInd], memUsage);
        }
      } // ENDFOR: dimInd
    }   // ENDFOR: bufInd
  }     // ENDFOR: layerInd
}

void CaffeEva::PrepCtrdBuf(void)
{
  // determine the size of each layer's centroid buffer
  ctrdBufStrLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    // obtain reference to the current layer's parameters
    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];

    // only allocate centroid buffer for the convolutional layers
    if ((layerInfo.type == ENUM_LyrType::Conv) ||
        (layerInfo.type == ENUM_LyrType::FCnt))
    {
      // obtain basic variables
      int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
      int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
      int featCntPerSpace = layerPara.ctrdLst.GetDimLen(2);

      // create the centroid buffer
      CtrdBufStr &ctrdBufStr = ctrdBufStrLst[layerInd];
      ctrdBufStr.dimCnt = 3;
      ctrdBufStr.dimLenLst[0] = subSpaceCnt;
      ctrdBufStr.dimLenLst[1] = featCntPerSpace;
      ctrdBufStr.dimLenLst[2] = ctrdCntPerSpace;
      std::shared_ptr<Matrix<float>> ptrCentroid(new Matrix<float>(layerPara.ctrdLst));
      ctrdBufStr.pCtrdBuf = ptrCentroid;
      ctrdBufStr.pCtrdBuf->Permute(0, 2, 1);
    } // ENDIF: layerInfo
  }   // ENDFOR: layerInd
}

void CaffeEva::PrepAsmtBuf(void)
{
  // determine the size of each layer's assignment buffer
  asmtBufStrLst.resize(caffeParaObj.layerCnt);
  for (int layerInd = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {
    // obtain reference to the current layer's parameters
    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];

    // allocate assignment buffer for the convolutional layer
    if (layerInfo.type == ENUM_LyrType::Conv)
    {
      // obtain basic variables
      int knlCnt = layerPara.asmtLst.GetDimLen(0);
      int knlHei = layerPara.asmtLst.GetDimLen(1);
      int knlWid = layerPara.asmtLst.GetDimLen(2);
      int subSpaceCnt = layerPara.asmtLst.GetDimLen(3);

      // create the assignment buffer
      AsmtBufStr &asmtBufStr = asmtBufStrLst[layerInd];
      asmtBufStr.dimCnt = 4;
      asmtBufStr.dimLenLst[0] = knlHei;
      asmtBufStr.dimLenLst[1] = knlWid;
      asmtBufStr.dimLenLst[2] = subSpaceCnt;
      asmtBufStr.dimLenLst[3] = knlCnt;
      std::shared_ptr<Matrix<uint8_t>> ptrAsm(new Matrix<uint8_t>(layerPara.asmtLst));
      asmtBufStr.pAsmtBuf = ptrAsm;
      asmtBufStr.pAsmtBuf->Permute(1, 2, 3, 0);

      // create the extended assignment buffer
      std::shared_ptr<Matrix<CBLAS_INT>> ptrAsmBuf(
          new Matrix<CBLAS_INT>(knlHei, knlWid, subSpaceCnt, knlCnt));
      asmtBufStr.pAsmtBufExt = ptrAsmBuf;
      const int eleCnt = asmtBufStr.pAsmtBuf->GetEleCnt();
      const uint8_t *asmtVecSrc = asmtBufStr.pAsmtBuf->GetDataPtr();
      CBLAS_INT *asmtVecDst = asmtBufStr.pAsmtBufExt->GetDataPtr();
      for (int eleInd = 0; eleInd < eleCnt; eleInd++)
      {
        asmtVecDst[eleInd] = asmtVecSrc[eleInd];
      } // ENDFOR: eleInd
    }   // ENDIF: layerInfo

    // allocate assignment buffer for the fully-connected layer
    if (layerInfo.type == ENUM_LyrType::FCnt)
    {
      // obtain basic variables
      int imgChnDst = layerPara.asmtLst.GetDimLen(0);
      int subSpaceCnt = layerPara.asmtLst.GetDimLen(1);

      // create the assignment buffer
      AsmtBufStr &asmtBufStr = asmtBufStrLst[layerInd];
      asmtBufStr.dimCnt = 2;
      asmtBufStr.dimLenLst[0] = subSpaceCnt;
      asmtBufStr.dimLenLst[1] = imgChnDst;
      std::shared_ptr<Matrix<uint8_t>> ptrAsm(new Matrix<uint8_t>(layerPara.asmtLst));
      asmtBufStr.pAsmtBuf = ptrAsm;
      asmtBufStr.pAsmtBuf->Permute(1, 0);
      //if(layerInd == 9)
      //  asmtBufStr.pAsmtBuf->Permute(1,0);

      // create the extended assignment buffer
      std::shared_ptr<Matrix<CBLAS_INT>> ptrAsmBuf(new Matrix<CBLAS_INT>(subSpaceCnt, imgChnDst));
      asmtBufStr.pAsmtBufExt = ptrAsmBuf;
      const int eleCnt = asmtBufStr.pAsmtBuf->GetEleCnt();
      const uint8_t *asmtVecSrc = asmtBufStr.pAsmtBuf->GetDataPtr();
      CBLAS_INT *asmtVecDst = asmtBufStr.pAsmtBufExt->GetDataPtr();
      for (int eleInd = 0; eleInd < eleCnt; eleInd++)
      {
        asmtVecDst[eleInd] = asmtVecSrc[eleInd];
      } // ENDFOR: eleInd
    }   // ENDIF: layerInfo
  }     // ENDFOR: layerInd
}

void CaffeEva::CalcFeatMap(const Matrix<float> &featMapSrc,
                           const int layerInd, Matrix<float> *pFeatMapDst)
{
  // determine the corresponding function for the current layer
  swAllLayers.Resume();
  switch (caffeParaObj.layerInfoLst[layerInd].type)
  {
  case ENUM_LyrType::Conv:
    swConvLayer.Resume();
    CalcFeatMap_Conv(featMapSrc, layerInd, pFeatMapDst);
    swConvLayer.Pause();
    break;
  case ENUM_LyrType::Pool:
    swPoolLayer.Resume();
    CalcFeatMap_Pool(featMapSrc, layerInd, pFeatMapDst);
    swPoolLayer.Pause();
    break;
  case ENUM_LyrType::AvgPool:
    swAvgPoolLayer.Resume();
    CalcFeatMap_AvgPool(featMapSrc, layerInd, pFeatMapDst);
    swAvgPoolLayer.Pause();
    break;
  case ENUM_LyrType::FCnt:
    swFCntLayer.Resume();
    CalcFeatMap_FCnt(featMapSrc, layerInd, pFeatMapDst);
    swFCntLayer.Pause();
    break;
  case ENUM_LyrType::ReLU:
    swReLuLayer.Resume();
    CalcFeatMap_ReLu(featMapSrc, layerInd, pFeatMapDst);
    swReLuLayer.Pause();
    break;
  case ENUM_LyrType::PReLU:
    swPReLULayer.Resume();
    CalcFeatMap_PReLU(featMapSrc, layerInd, pFeatMapDst);
    swPReLULayer.Pause();
    break;
  case ENUM_LyrType::LoRN:
    if (caffeParaObj.layerInfoLst[layerInd].lrnAcr)
    {
      swLoRNLayer.Resume();
      CalcFeatMap_LoRN(featMapSrc, layerInd, pFeatMapDst);
      swLoRNLayer.Pause();
    }
    else
    {
      swLoRNLayer.Resume();
      CalcFeatMap_LoRNWithin(featMapSrc, layerInd, pFeatMapDst);
      swLoRNLayer.Pause();
    }
    break;
  case ENUM_LyrType::Drpt:
    swDrptLayer.Resume();
    CalcFeatMap_Drpt(featMapSrc, layerInd, pFeatMapDst);
    swDrptLayer.Pause();
    break;
  case ENUM_LyrType::SMax:
    swSMaxLayer.Resume();
    CalcFeatMap_SMax(featMapSrc, layerInd, pFeatMapDst);
    swSMaxLayer.Pause();
    break;
  default:
    printf("[ERROR] invalid layer type\n");
    return;
  } // ENDSWITCH: caffeParaObj
  swAllLayers.Pause();
}

void CaffeEva::CalcFeatMap_Conv(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  if (enblAprx)
  {
    CalcFeatMap_ConvAprx2(featMapSrc, layerInd, pFeatMapDst);
  }
  else
  {
    CalcFeatMap_ConvPrec(featMapSrc, layerInd, pFeatMapDst);
  } // ENDIF: enblAprx
}

void CaffeEva::CalcFeatMap_ConvPrec(const Matrix<float> &featMapSrc,
                                    const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  int knlCnt = layerPara.convKnlLst.GetDimLen(0);
  int knlSiz = layerPara.convKnlLst.GetDimLen(2);
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChnSrc = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);
  int imgChnDst = pFeatMapDst->GetDimLen(3);
  int knlCntPerGrp = knlCnt / layerInfo.grpCnt;
  int imgChnSrcPerGrp = imgChnSrc / layerInfo.grpCnt;

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float> &featMapSrcPrm = *(featBufStrMat[layerInd][0].pFeatBuf);
  Matrix<float> &featMapSrcRsp = *(featBufStrMat[layerInd][1].pFeatBuf);
  Matrix<float> &featMapDstRsp = *(featBufStrMat[layerInd][2].pFeatBuf);

  // permute the input feature map dimensions
  featMapSrcPrm.Resize(dataCnt, imgHeiSrc, imgWidSrc, imgChnSrc);
  memcpy(featMapSrcPrm.GetDataPtr(),
         featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
  featMapSrcPrm.Permute(0, 3, 1, 2);

  // reshape the output feature map
  pFeatMapDst->Resize(dataCnt, imgChnDst, imgHeiDst, imgWidDst);

  // compute the feature map after passing a convolutional layer
  const float *biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    for (int grpInd = 0; grpInd < layerInfo.grpCnt; grpInd++)
    {
      // copy source feature map to feature buffer
      // im2col or convolution unrolling
      CvtFeatMapToFeatBuf(
          featMapSrcPrm, dataInd, grpInd, layerInfo, &featMapSrcRsp);

      // call CBLAS function to compute the matrix-matrix multiplication
      int knlIndL = grpInd * knlCntPerGrp;
      CBLAS_ORDER order = CblasRowMajor;
      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;
      CBLAS_INT m = knlCntPerGrp;
      CBLAS_INT n = imgHeiDst * imgWidDst;
      CBLAS_INT k = imgChnSrcPerGrp * knlSiz * knlSiz;
      CBLAS_INT lda = k;
      CBLAS_INT ldb = n;
      CBLAS_INT ldc = n;
      float alpha = 1.0;
      float beta = 0.0;
      float *pa = layerPara.convKnlLst.GetDataPtr(knlIndL, 0, 0, 0);
      float *pb = featMapSrcRsp.GetDataPtr();
      float *pc = featMapDstRsp.GetDataPtr();
      cblas_sgemm(order, transA, transB,
                  m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);

      // append the bias term
      int rowCntBuf = featMapDstRsp.GetDimLen(0);
      int colCntBuf = featMapDstRsp.GetDimLen(1);
      for (int rowIndBuf = 0; rowIndBuf < rowCntBuf; rowIndBuf++)
      {
        const float biasVal = biasVec[rowIndBuf + grpInd * knlCntPerGrp];
        float *pFeatVecDstRsp = featMapDstRsp.GetDataPtr(rowIndBuf, 0);
        for (int colIndBuf = 0; colIndBuf < colCntBuf; colIndBuf++)
        {
          pFeatVecDstRsp[colIndBuf] += biasVal;
        } // ENDFOR: colIndBuf
      }   // ENDFOR: rowIndBuf

      // copy feature buffer to target feature map
      CvtFeatBufToFeatMap(
          featMapDstRsp, dataInd, grpInd, layerInfo, pFeatMapDst);
    } // ENDFOR: grpInd
  }   // ENDFOR: dataInd

  // permute the output feature map dimensions
  pFeatMapDst->Permute(0, 2, 3, 1);
}

void CaffeEva::CalcFeatMap_ConvAprx(const Matrix<float> &featMapSrc,
                                    const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  int knlCnt = layerInfo.knlCnt;
  int knlHei = layerInfo.knlSiz;
  int knlWid = layerInfo.knlSiz;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChnSrc = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);
  int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
  int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
  int ctrdCntExt = ctrdCntPerSpace * subSpaceCnt;

  // determine the size of feature map groups
  int knlCntPerGrp = knlCnt / layerInfo.grpCnt;
  int imgChnSrcPerGrp = imgChnSrc / layerInfo.grpCnt;

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float> &featMapSrcPerGrp = *(featBufStrMat[layerInd][3].pFeatBuf);
  Matrix<float> &inPdMat = *(featBufStrMat[layerInd][4].pFeatBuf);

  // obtain pre-allocated centroid and assignment buffer
  Matrix<float> &ctrdBuf = *(ctrdBufStrLst[layerInd].pCtrdBuf);
  Matrix<uint8_t> &asmtBuf = *(asmtBufStrLst[layerInd].pAsmtBuf);
  // Matrix<CBLAS_INT>& asmtBufExt = *(asmtBufStrLst[layerInd].pAsmtBufExt);

  // compute the feature map after passing a convolutional layer
  int sptCntDst = imgHeiDst * imgWidDst;
  // int sptCntKnl = knlHei * knlWid;
  const float *biasVec = layerPara.biasVec.GetDataPtr();
  for (int grpInd = 0; grpInd < layerInfo.grpCnt; grpInd++)
  {
    // obtain basic variables for the current feature map group
    int knlIndL = knlCntPerGrp * grpInd;
    int chnIndSrcL = imgChnSrcPerGrp * grpInd;

    // quantize the source feature map with pre-defined codebook
    swCompLkupTblConv.Resume();
    featMapSrcPerGrp.Resize(dataCnt, imgHeiSrc, imgWidSrc, imgChnSrcPerGrp);
    if (layerInfo.grpCnt == 1)
    {
      memcpy(featMapSrcPerGrp.GetDataPtr(),
             featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
    }
    else
    {
      featMapSrc.GetSubMat(0, 0, 0, chnIndSrcL, &featMapSrcPerGrp);
    } // ENDIF: layerInfo
    featMapSrcPerGrp.Resize(dataCnt * imgHeiSrc * imgWidSrc, imgChnSrcPerGrp);
    GetInPdMat(featMapSrcPerGrp, ctrdBuf, &inPdMat);
    inPdMat.Resize(dataCnt, imgHeiSrc, imgWidSrc, ctrdCntExt);
    swCompLkupTblConv.Pause();

    // compute the target response via table look-up operations
    swEstiInPdValConv.Resume();
    for (int sptIndDst = 0; sptIndDst < sptCntDst; sptIndDst++)
    {
      // determine the corresponding indexes in the target/source feature map
      int heiIndDst = sptIndDst / imgWidDst;
      int widIndDst = sptIndDst % imgWidDst;
      int heiIndSrcL = heiIndDst * layerInfo.stride - layerInfo.padSiz;
      int widIndSrcL = widIndDst * layerInfo.stride - layerInfo.padSiz;

      // determine the lower/upper bound in the convolutional kernels
      int heiIndKnlL = std::max(0, 0 - heiIndSrcL);
      int heiIndKnlU = std::min(knlHei - 1, imgHeiSrc - 1 - heiIndSrcL);
      int widIndKnlL = std::max(0, 0 - widIndSrcL);
      int widIndKnlU = std::min(knlWid - 1, imgWidSrc - 1 - widIndSrcL);

      // compute the target feature map for each instance
      for (int dataInd = 0; dataInd < dataCnt; dataInd++)
      {
        // initialize <convSumLst> with the bias term
        float *featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, knlIndL);
        memcpy(featVecDst, biasVec + knlIndL, sizeof(float) * knlCntPerGrp);

        // compute the target response via table look-up operations
        int heiIndKnl;   // to shorten the line length
        int widIndKnl;   // to shorten the line length
        int subSpaceInd; // to shorten the line length
        for (heiIndKnl = heiIndKnlL; heiIndKnl <= heiIndKnlU; heiIndKnl++)
        {
          for (widIndKnl = widIndKnlL; widIndKnl <= widIndKnlU; widIndKnl++)
          {
            int heiIndSrc = heiIndSrcL + heiIndKnl;
            int widIndSrc = widIndSrcL + widIndKnl;
            const float *inPdVec =
                inPdMat.GetDataPtr(dataInd, heiIndSrc, widIndSrc, 0);
            const uint8_t *asmtVec =
                asmtBuf.GetDataPtr(heiIndKnl, widIndKnl, 0, knlIndL);
            for (subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++)
            {
              for (int knlInd = 0; knlInd < knlCntPerGrp; knlInd += 8)
              {
                featVecDst[knlInd] += inPdVec[asmtVec[knlInd]];
                featVecDst[knlInd + 1] += inPdVec[asmtVec[knlInd + 1]];
                featVecDst[knlInd + 2] += inPdVec[asmtVec[knlInd + 2]];
                featVecDst[knlInd + 3] += inPdVec[asmtVec[knlInd + 3]];
                featVecDst[knlInd + 4] += inPdVec[asmtVec[knlInd + 4]];
                featVecDst[knlInd + 5] += inPdVec[asmtVec[knlInd + 5]];
                featVecDst[knlInd + 6] += inPdVec[asmtVec[knlInd + 6]];
                featVecDst[knlInd + 7] += inPdVec[asmtVec[knlInd + 7]];
              } // ENDFOR: knlInd
              inPdVec += ctrdCntPerSpace;
              asmtVec += knlCnt;
            } // ENDFOR: subSpaceInd
          }   // ENDFOR: widIndKnl
        }     // ENDFOR: heiIndKnl
      }       // ENDFOR: dataInd
    }         // ENDFOR: sptIndDst
    swEstiInPdValConv.Pause();
  } // ENDFOR: grpInd
}

// Modified: 2018 May
// Programer:  Yi-Ming Chan
// Take the row(width) index of the inner loop
void CaffeEva::CalcFeatMap_ConvAprx2(const Matrix<float> &featMapSrc,
                                     const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  int knlCnt = layerInfo.knlCnt;
  int knlHei = layerInfo.knlSiz;
  int knlWid = layerInfo.knlSiz;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChnSrc = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);
  int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
  int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
  int ctrdCntExt = ctrdCntPerSpace * subSpaceCnt;

  // determine the size of feature map groups
  int knlCntPerGrp = knlCnt / layerInfo.grpCnt;
  int imgChnSrcPerGrp = imgChnSrc / layerInfo.grpCnt;

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float> &featMapSrcPerGrp = *(featBufStrMat[layerInd][3].pFeatBuf);
  Matrix<float> &inPdMat = *(featBufStrMat[layerInd][4].pFeatBuf);

  // obtain pre-allocated centroid and assignment buffer
  Matrix<float> &ctrdBuf = *(ctrdBufStrLst[layerInd].pCtrdBuf);
  Matrix<uint8_t> &asmtBuf = *(asmtBufStrLst[layerInd].pAsmtBuf);
  // Matrix<CBLAS_INT>& asmtBufExt = *(asmtBufStrLst[layerInd].pAsmtBufExt);

  // compute the feature map after passing a convolutional layer
  int sptCntDst = imgHeiDst * imgWidDst;
  // int sptCntKnl = knlHei * knlWid;
  const float *biasVec = layerPara.biasVec.GetDataPtr();
  for (int grpInd = 0; grpInd < layerInfo.grpCnt; grpInd++)
  {
    // obtain basic variables for the current feature map group
    int knlIndL = knlCntPerGrp * grpInd;
    int chnIndSrcL = imgChnSrcPerGrp * grpInd;

    // quantize the source feature map with pre-defined codebook
    swCompLkupTblConv.Resume();
    featMapSrcPerGrp.Resize(dataCnt, imgHeiSrc, imgWidSrc, imgChnSrcPerGrp);
    if (layerInfo.grpCnt == 1)
    {
      memcpy(featMapSrcPerGrp.GetDataPtr(),
             featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
    }
    else
    {
      featMapSrc.GetSubMat(0, 0, 0, chnIndSrcL, &featMapSrcPerGrp);
    } // ENDIF: layerInfo
    featMapSrcPerGrp.Resize(dataCnt * imgHeiSrc * imgWidSrc, imgChnSrcPerGrp);
    GetInPdMat(featMapSrcPerGrp, ctrdBuf, &inPdMat);
    inPdMat.Resize(dataCnt, imgHeiSrc, imgWidSrc, ctrdCntExt);
    swCompLkupTblConv.Pause();

    // compute the target response via table look-up operations
    swEstiInPdValConv.Resume();
    for (int sptIndDst = 0; sptIndDst < sptCntDst; sptIndDst++)
    {
      // determine the corresponding indexes in the target/source feature map
      int heiIndDst = sptIndDst / imgWidDst;
      int widIndDst = sptIndDst % imgWidDst;
      int heiIndSrcL = heiIndDst * layerInfo.stride - layerInfo.padSiz;
      int widIndSrcL = widIndDst * layerInfo.stride - layerInfo.padSiz;

      // determine the lower/upper bound in the convolutional kernels
      int heiIndKnlL = std::max(0, 0 - heiIndSrcL);
      int heiIndKnlU = std::min(knlHei - 1, imgHeiSrc - 1 - heiIndSrcL);
      int widIndKnlL = std::max(0, 0 - widIndSrcL);
      int widIndKnlU = std::min(knlWid - 1, imgWidSrc - 1 - widIndSrcL);

      // compute the target feature map for each instance
      for (int dataInd = 0; dataInd < dataCnt; dataInd++)
      {
        // initialize <convSumLst> with the bias term
        float *featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, knlIndL);
        memcpy(featVecDst, biasVec + knlIndL, sizeof(float) * knlCntPerGrp);

        // compute the target response via table look-up operations
        int heiIndKnl;   // to shorten the line length
        int widIndKnl;   // to shorten the line length
        int subSpaceInd; // to shorten the line length
        for (heiIndKnl = heiIndKnlL; heiIndKnl <= heiIndKnlU; heiIndKnl++)
        {
          int heiIndSrc = heiIndSrcL + heiIndKnl;
          float *inPdVecRow =
              inPdMat.GetDataPtr(dataInd, heiIndSrc, widIndSrcL, 0);

          for (widIndKnl = widIndKnlL; widIndKnl <= widIndKnlU; widIndKnl++)
          {
            int widIndSrc = widIndSrcL + widIndKnl;
            float *inPdVec = inPdVecRow + widIndKnl * inPdMat.GetDimLen(3);
            const uint8_t *asmtVec =
                asmtBuf.GetDataPtr(heiIndKnl, widIndKnl, 0, knlIndL);
            for (subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++)
            {
              for (int knlInd = 0; knlInd < knlCntPerGrp; knlInd += 8)
              {
                featVecDst[knlInd] += inPdVec[asmtVec[knlInd]];
                featVecDst[knlInd + 1] += inPdVec[asmtVec[knlInd + 1]];
                featVecDst[knlInd + 2] += inPdVec[asmtVec[knlInd + 2]];
                featVecDst[knlInd + 3] += inPdVec[asmtVec[knlInd + 3]];
                featVecDst[knlInd + 4] += inPdVec[asmtVec[knlInd + 4]];
                featVecDst[knlInd + 5] += inPdVec[asmtVec[knlInd + 5]];
                featVecDst[knlInd + 6] += inPdVec[asmtVec[knlInd + 6]];
                featVecDst[knlInd + 7] += inPdVec[asmtVec[knlInd + 7]];
              } // ENDFOR: knlInd
              inPdVec += ctrdCntPerSpace;
              asmtVec += knlCnt;
            } // ENDFOR: subSpaceInd
          }   // ENDFOR: widIndKnl
        }     // ENDFOR: heiIndKnl
      }       // ENDFOR: dataInd
    }         // ENDFOR: sptIndDst
    swEstiInPdValConv.Pause();
  } // ENDFOR: grpInd
}

void CaffeEva::CalcFeatMap_Pool(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int padSiz = layerInfo.padSiz;
  int knlSiz = layerInfo.knlSiz;
  int stride = layerInfo.stride;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);

  // compute the feature map after passing a convolutional layer
  for (int heiIndDst = 0; heiIndDst < imgHeiDst; heiIndDst++)
  {
    // determine the corresponding indexes in the source feature map
    int heiIndSrcL = std::max(0, heiIndDst * stride - padSiz);
    int heiIndSrcU =
        std::min(imgHeiSrc, heiIndDst * stride + knlSiz - padSiz) - 1;
    int heiCntSrcSel = heiIndSrcU - heiIndSrcL + 1;

    for (int widIndDst = 0; widIndDst < imgWidDst; widIndDst++)
    {
      // determine the corresponding indexes in the source feature map
      int widIndSrcL = std::max(0, widIndDst * stride - padSiz);
      int widIndSrcU =
          std::min(imgWidSrc, widIndDst * stride + knlSiz - padSiz) - 1;
      int widCntSrcSel = widIndSrcU - widIndSrcL + 1;
      int sptCntSrcSel = heiCntSrcSel * widCntSrcSel;

      // perform max-pooling operation
      for (int dataInd = 0; dataInd < dataCnt; dataInd++)
      {
        float *featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, 0);
        for (int sptIndSrc = 0; sptIndSrc < sptCntSrcSel; sptIndSrc++)
        {
          int heiIndSrc = heiIndSrcL + sptIndSrc / widCntSrcSel;
          int widIndSrc = widIndSrcL + sptIndSrc % widCntSrcSel;
          const float *featVecSrc =
              featMapSrc.GetDataPtr(dataInd, heiIndSrc, widIndSrc, 0);
          if (sptIndSrc == 0)
          {
            memcpy(featVecDst, featVecSrc, sizeof(float) * imgChn);
          }
          else
          {
            for (int chnInd = 0; chnInd < imgChn; chnInd++)
            {
              featVecDst[chnInd] =
                  std::max(featVecSrc[chnInd], featVecDst[chnInd]);
            } // ENDFOR: chnInd
          }   // ENDIF: sptIndSrc
        }     // ENDFOR: sptIndSrc
      }       // ENDFOR: dataInd
    }         // ENDFOR: widIndDst
  }           // ENDFOR: heiIndDst
}

void CaffeEva::CalcFeatMap_AvgPool(const Matrix<float> &featMapSrc,
                                   const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int padSiz = layerInfo.padSiz;
  int knlSiz = layerInfo.knlSiz;
  int stride = layerInfo.stride;
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  int imgHeiDst = pFeatMapDst->GetDimLen(1);
  int imgWidDst = pFeatMapDst->GetDimLen(2);

  // compute the feature map after passing a convolutional layer
  for (int heiIndDst = 0; heiIndDst < imgHeiDst; heiIndDst++)
  {
    // determine the corresponding indexes in the source feature map
    int heiIndSrcL = std::max(0, heiIndDst * stride - padSiz);
    int heiIndSrcU =
        std::min(imgHeiSrc, heiIndDst * stride + knlSiz - padSiz) - 1;
    int heiCntSrcSel = heiIndSrcU - heiIndSrcL + 1;

    for (int widIndDst = 0; widIndDst < imgWidDst; widIndDst++)
    {
      // determine the corresponding indexes in the source feature map
      int widIndSrcL = std::max(0, widIndDst * stride - padSiz);
      int widIndSrcU =
          std::min(imgWidSrc, widIndDst * stride + knlSiz - padSiz) - 1;
      int widCntSrcSel = widIndSrcU - widIndSrcL + 1;
      int sptCntSrcSel = heiCntSrcSel * widCntSrcSel;

      // perform average-pooling operation
      for (int dataInd = 0; dataInd < dataCnt; dataInd++)
      {
        float *featVecDst =
            pFeatMapDst->GetDataPtr(dataInd, heiIndDst, widIndDst, 0);

        float *featSum = new float[imgChn](); // Initializing to zero
        for (int heiIndSrc = heiIndSrcL; heiIndSrc <= heiIndSrcU; heiIndSrc++)
        {
          for (int widIndSrc = widIndSrcL; widIndSrc <= widIndSrcU; widIndSrc++)
          {

            const float *featVecSrc =
                featMapSrc.GetDataPtr(dataInd, heiIndSrc, widIndSrc, 0);
            for (int chnInd = 0; chnInd < imgChn; chnInd++)
            {
              featSum[chnInd] = featSum[chnInd] + featVecSrc[chnInd];
            } // ENDFOR: chnInd

          } // ENDFOR: widIndSrc
        }   // ENDFOR: heiIndSrc

        for (int chnInd = 0; chnInd < imgChn; chnInd++)
        {
          featVecDst[chnInd] = featSum[chnInd] / sptCntSrcSel;
        } // ENDFOR: chnInd

        delete[] featSum;

      } // ENDFOR: dataInd
    }   // ENDFOR: widIndDst
  }     // ENDFOR: heiIndDst
}

void CaffeEva::CalcFeatMap_FCnt(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  if (enblAprx)
  {
    CalcFeatMap_FCntAprx(featMapSrc, layerInd, pFeatMapDst);
  }
  else
  {
    CalcFeatMap_FCntPrec(featMapSrc, layerInd, pFeatMapDst);
  } // ENDIF: enblAprx
}

void CaffeEva::CalcFeatMap_FCntPrec(const Matrix<float> &featMapSrc,
                                    const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChnSrc = featMapSrc.GetDimStp(0);
  int imgChnDst = pFeatMapDst->GetDimStp(0);

  // call CBLAS function to compute the matrix-matrix multiplication
  CBLAS_ORDER order = CblasRowMajor;
  CBLAS_TRANSPOSE transA = CblasNoTrans;
  CBLAS_TRANSPOSE transB = CblasTrans;
  CBLAS_INT m = dataCnt;
  CBLAS_INT n = imgChnDst;
  CBLAS_INT k = imgChnSrc;
  CBLAS_INT lda = k;
  CBLAS_INT ldb = k;
  CBLAS_INT ldc = n;
  float alpha = 1.0;
  float beta = 0.0;
  float *pa = featMapSrc.GetDataPtr();
  float *pb = layerPara.fcntWeiMat.GetDataPtr();
  float *pc = pFeatMapDst->GetDataPtr();
  cblas_sgemm(order, transA, transB,
              m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);

  // append the bias term
  const float *biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    float *pFeatMapVec = pFeatMapDst->GetDataPtr(dataInd, 0);
    for (int chnIndDst = 0; chnIndDst < imgChnDst; chnIndDst++)
    {
      pFeatMapVec[chnIndDst] += biasVec[chnIndDst];
    } // ENDFOR: chnIndDst
  }   // ENDFOR: dataInd
}

void CaffeEva::CalcFeatMap_FCntAprx(const Matrix<float> &featMapSrc,
                                    const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChnDst = pFeatMapDst->GetDimStp(0);
  int subSpaceCnt = layerPara.ctrdLst.GetDimLen(0);
  int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);

  // obtain pre-allocated matrices for auxiliary variables
  Matrix<float> &featMapSrcRsp = *(featBufStrMat[layerInd][0].pFeatBuf);
  Matrix<float> &inPdMat = *(featBufStrMat[layerInd][1].pFeatBuf);
  inPdMat.Resize(dataCnt, subSpaceCnt, ctrdCntPerSpace);

  // obtain pre-allocated centroid and assignment buffer
  Matrix<float> &ctrdBuf = *(ctrdBufStrLst[layerInd].pCtrdBuf);
  Matrix<uint8_t> &asmtBuf = *(asmtBufStrLst[layerInd].pAsmtBuf);
  // Matrix<CBLAS_INT>& asmtBufExt = *(asmtBufStrLst[layerInd].pAsmtBufExt);

  // quantize the source feature map with pre-defined codebook
  swCompLkupTblFCnt.Resume();
  memcpy(featMapSrcRsp.GetDataPtr(),
         featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
  GetInPdMat(featMapSrcRsp, ctrdBuf, &inPdMat);
  inPdMat.Resize(dataCnt, subSpaceCnt * ctrdCntPerSpace);
  swCompLkupTblFCnt.Pause();

  // compute the feature map after passing a fully-connected layer
  swEstiInPdValFCnt.Resume();
  const float *biasVec = layerPara.biasVec.GetDataPtr();
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    // initialize target response with the bias term
    float *featVecDst = pFeatMapDst->GetDataPtr(dataInd, 0, 0, 0);
    memcpy(featVecDst, biasVec, sizeof(float) * imgChnDst);

    // update target response with look-up operations
    const float *inPdVec = inPdMat.GetDataPtr(dataInd, 0); // index offset
    const uint8_t *asmtVec = asmtBuf.GetDataPtr();
    for (int subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++)
    {
      // update the target response within the current subspace
      for (int chnIndDst = 0; chnIndDst < imgChnDst; chnIndDst += 1)
      {
        featVecDst[chnIndDst] += inPdVec[asmtVec[chnIndDst]];
        //featVecDst[chnIndDst + 1] += inPdVec[asmtVec[chnIndDst + 1]];
        //featVecDst[chnIndDst + 2] += inPdVec[asmtVec[chnIndDst + 2]];
        //featVecDst[chnIndDst + 3] += inPdVec[asmtVec[chnIndDst + 3]];
        //featVecDst[chnIndDst + 4] += inPdVec[asmtVec[chnIndDst + 4]];
        //featVecDst[chnIndDst + 5] += inPdVec[asmtVec[chnIndDst + 5]];
        //featVecDst[chnIndDst + 6] += inPdVec[asmtVec[chnIndDst + 6]];
        //featVecDst[chnIndDst + 7] += inPdVec[asmtVec[chnIndDst + 7]];
      } // ENDFOR: chnIndDst

      // update pointers to the look-up table and assignment variable
      inPdVec += ctrdCntPerSpace;
      asmtVec += imgChnDst;
    } // ENDFOR: subSpaceInd
  }   // ENDFOR: dataInd
  swEstiInPdValFCnt.Pause();
}

void CaffeEva::CalcFeatMap_ReLu(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  // compute the feature map after passing a ReLu layer
  int eleCnt = featMapSrc.GetEleCnt();
  const float *featVecSrc = featMapSrc.GetDataPtr();
  float *featVecDst = pFeatMapDst->GetDataPtr();
  for (int eleInd = 0; eleInd < eleCnt; eleInd++)
  {
    featVecDst[eleInd] = std::max(0.0f, featVecSrc[eleInd]);
  } // ENDFOR: eleInd
}

void CaffeEva::CalcFeatMap_Shortcut(const Matrix<float> &featMapSrc,
                                    const Matrix<float> &featMapFrom,
                                    Matrix<float> *pFeatMapDst)
{
  // compute the feature map after passing a ReLu layer
  int eleCnt = featMapSrc.GetEleCnt();
  const float *featVecSrc = featMapSrc.GetDataPtr();
  const float *featVecFrom = featMapFrom.GetDataPtr();
  float *featVecDst = pFeatMapDst->GetDataPtr();

  vsAdd(eleCnt, featVecSrc, featVecFrom, featVecDst);
}

// Todo implement
void CaffeEva::CalcFeatMap_PReLU(const Matrix<float> &featMapSrc,
                                 const int layerInd, Matrix<float> *pFeatMapDst)
{
  // compute the feature map after passing a PReLu layer
  int eleCnt = featMapSrc.GetEleCnt();
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHeiSrc = featMapSrc.GetDimLen(1);
  int imgWidSrc = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);  
  const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
  const float *featVecSrc = featMapSrc.GetDataPtr();

  float *featVecDst = pFeatMapDst->GetDataPtr();
  for (int eleInd = 0; eleInd < eleCnt; eleInd++)
  {
    int c = eleInd % imgChn; // Source array is [data, H, W, C]
    featVecDst[eleInd] = std::max(0.0f, featVecSrc[eleInd]) 
      + layerPara.alphaVec.GetEleAt(c) * std::min(0.0f, featVecSrc[eleInd]);
  } // ENDFOR: eleInd
}

void CaffeEva::CalcFeatMap_LoRN(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHei = featMapSrc.GetDimLen(1);
  int imgWid = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  int lrnRad = (layerInfo.lrnSiz - 1) / 2;
  int sptCnt = imgHei * imgWid;

  // declare auxiliary variable arrays
  int imgChnExt = imgChn + lrnRad * 2;
  float *featVecSrcExt = new float[imgChnExt];
  float *loclSumLst = new float[imgChn];

  // compute the feature map after passing a local response normalization layer
  float coeffVal = layerInfo.lrnAlp / layerInfo.lrnSiz;
  memset(featVecSrcExt, 0, sizeof(float) * imgChnExt);
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    for (int sptInd = 0; sptInd < sptCnt; sptInd++)
    {
      // determine the height/width indexes
      int heiInd = sptInd / imgWid;
      int widInd = sptInd % imgWid;

      // compute the squared feature vector
      const float *featVecSrc =
          featMapSrc.GetDataPtr(dataInd, heiInd, widInd, 0);
      vsSqr(imgChn, featVecSrc, featVecSrcExt + lrnRad);
      cblas_sscal(imgChn, coeffVal, featVecSrcExt + lrnRad, 1);

      // compute <loclSumLst> with a sliding windows
      for (int chnInd = 0; chnInd < imgChn; chnInd++)
      {
        loclSumLst[chnInd] = layerInfo.lrnIni;
      } // ENDFOR: chnInd
      for (int chnInd = 0; chnInd < layerInfo.lrnSiz; chnInd++)
      {
        vsAdd(imgChn, loclSumLst, featVecSrcExt + chnInd, loclSumLst);
      } // ENDFOR: chnInd

      // transform local patch sum to normalization factor
      vsPowx_m(imgChn, loclSumLst, -layerInfo.lrnBet, loclSumLst);

      // compute the normalized feature map
      float *featVecDst = pFeatMapDst->GetDataPtr(dataInd, heiInd, widInd, 0);
      vsMul(imgChn, featVecSrc, loclSumLst, featVecDst);
    } // ENDFOR: sptInd
  }   // ENDFOR: dataInd

  // release auxiliary variable arrays
  delete[] featVecSrcExt;
  delete[] loclSumLst;
}

void CaffeEva::CalcFeatMap_LoRNWithin(const Matrix<float> &featMapSrc,
                                      const int layerInd, Matrix<float> *pFeatMapDst)
{
  // obtain basic variables
  const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgHei = featMapSrc.GetDimLen(1);
  int imgWid = featMapSrc.GetDimLen(2);
  int imgChn = featMapSrc.GetDimLen(3);
  Matrix<float> featMapSrcClone(featMapSrc);

  // Get the square of all input
  int total = dataCnt * imgHei * imgWid * imgChn;
  float *fpSrcClone = featMapSrcClone.GetDataPtr();
  for (size_t i = 0; i < total; i++)
  {
    fpSrcClone[i] = fpSrcClone[i] * fpSrcClone[i];
  }

  //featMapSrcClone.Permute(0,3,1,2); // Permute into Channel Height Width

  int lrnRad = (layerInfo.lrnSiz - 1) / 2;

  // declare auxiliary variable arrays
  float *loclSumLst = new float[imgChn];
  float *initialChn = new float[imgChn];

  for (size_t i = 0; i < imgChn; i++)
  {
    initialChn[i] = layerInfo.lrnIni;
  }

  // compute the feature map after passing a local response normalization layer
  float coeffVal = layerInfo.lrnAlp / (layerInfo.lrnSiz * layerInfo.lrnSiz);

  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    // Process each pixel
    for (int heiInd = 0; heiInd < imgHei; heiInd++)
    {
      // determine the corresponding indexes in the source feature map
      int heiIndSrcL = std::max(0, heiInd - lrnRad);
      int heiIndSrcU =
          std::min(imgHei - 1, heiInd + lrnRad);
      for (int widInd = 0; widInd < imgWid; widInd++)
      {
        // determine the corresponding indexes in the source feature map
        int widIndSrcL = std::max(0, widInd - lrnRad);
        int widIndSrcU =
            std::min(imgWid - 1, widInd + lrnRad);

        memset(loclSumLst, 0, sizeof(float) * imgChn);

        for (size_t iheiSrcInd = heiIndSrcL; iheiSrcInd <= heiIndSrcU; iheiSrcInd++)
        {
          for (size_t iwidSrcInd = widIndSrcL; iwidSrcInd <= widIndSrcU; iwidSrcInd++)
          {
            // Sum all
            float *featSrc = featMapSrcClone.GetDataPtr(dataInd, iheiSrcInd, iwidSrcInd, 0);
            vsAdd(imgChn, loclSumLst, featSrc, loclSumLst);
          } // ENDFOR: iwidSrcInd
        }   // ENDFOR: iheiSrcInd

        cblas_sscal(imgChn, coeffVal, loclSumLst, 1);
        vsAdd(imgChn, loclSumLst, initialChn, loclSumLst);

        // transform local patch sum to normalization factor
        vsPowx_m(imgChn, loclSumLst, -layerInfo.lrnBet, loclSumLst);

        const float *featVecSrc =
            featMapSrc.GetDataPtr(dataInd, heiInd, widInd, 0);

        // compute the normalized feature map
        float *featVecDst = pFeatMapDst->GetDataPtr(dataInd, heiInd, widInd, 0);
        vsMul(imgChn, featVecSrc, loclSumLst, featVecDst);

      } // ENDFOR: widInd
    }   // ENDFOR: heiInd
  }     // ENDFOR: dataInd

  // release auxiliary variable arrays
  delete[] loclSumLst;
  delete[] initialChn;
}

void CaffeEva::CalcFeatMap_Drpt(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  // compute the feature map after passing a dropout layer
  memcpy(pFeatMapDst->GetDataPtr(),
         featMapSrc.GetDataPtr(), sizeof(float) * featMapSrc.GetEleCnt());
}

void CaffeEva::CalcFeatMap_SMax(const Matrix<float> &featMapSrc,
                                const int layerInd, Matrix<float> *pFeatMapDst)
{
  // compute the feature map after passing a softmax layer
  int dataCnt = featMapSrc.GetDimLen(0);
  int imgChn = featMapSrc.GetDimStp(0);
  for (int dataInd = 0; dataInd < dataCnt; dataInd++)
  {
    const float *featVecSrc = featMapSrc.GetDataPtr(dataInd, 0, 0, 0);
    float *featVecDst = pFeatMapDst->GetDataPtr(dataInd, 0, 0, 0);

    double sum = 0.0;
    double tempExp[imgChn];
    for (int chnInd = 0; chnInd < imgChn; chnInd++)
    {
      tempExp[chnInd] = exp(featVecSrc[chnInd]);
      sum += tempExp[chnInd];
    } // ENDFOR: chnInd
    for (int chnInd = 0; chnInd < imgChn; chnInd++)
    {
      featVecDst[chnInd] = tempExp[chnInd] / sum;
    } // ENDFOR: chnInd
  }   // ENDFOR: dataInd
}

void CaffeEva::InitFeatBuf(
    FeatBufStr *pFeatBufStr, const ENUM_BufUsage us, const int d0)
{
  pFeatBufStr->usage = us;
  pFeatBufStr->dimCnt = 1;
  pFeatBufStr->dimLenLst[0] = d0;
}

void CaffeEva::InitFeatBuf(FeatBufStr *pFeatBufStr,
                           const ENUM_BufUsage us, const int d0, const int d1)
{
  InitFeatBuf(pFeatBufStr, us, d0);
  pFeatBufStr->dimCnt = 2;
  pFeatBufStr->dimLenLst[1] = d1;
}

void CaffeEva::InitFeatBuf(FeatBufStr *pFeatBufStr,
                           const ENUM_BufUsage us, const int d0, const int d1, const int d2)
{
  InitFeatBuf(pFeatBufStr, us, d0, d1);
  pFeatBufStr->dimCnt = 3;
  pFeatBufStr->dimLenLst[2] = d2;
}

void CaffeEva::InitFeatBuf(FeatBufStr *pFeatBufStr, const ENUM_BufUsage us,
                           const int d0, const int d1, const int d2, const int d3)
{
  InitFeatBuf(pFeatBufStr, us, d0, d1, d2);
  pFeatBufStr->dimCnt = 4;
  pFeatBufStr->dimLenLst[3] = d3;
}

void CaffeEva::CvtDataLstToFeatMap(const int dataIndL, const int dataIndU,
                                   const Matrix<float> &dataLst, Matrix<float> *pFeatMap)
{
  // obtain basic variables
  int dataVecLen = dataLst.GetDimStp(0);
  int imgChn = dataLst.GetDimLen(1);
  int imgHei = dataLst.GetDimLen(2);
  int imgWid = dataLst.GetDimLen(3);

  // copy each sample's feature vector
  int dataCntSel = dataIndU - dataIndL + 1;
  pFeatMap->Resize(dataCntSel, imgChn, imgHei, imgWid);
  memcpy(pFeatMap->GetDataPtr(), dataLst.GetDataPtr(dataIndL, 0, 0, 0),
         sizeof(float) * dataVecLen * dataCntSel);
  pFeatMap->Permute(0, 2, 3, 1);
}

void CaffeEva::CvtFeatMapToLablVec(const int dataIndL, const int dataIndU,
                                   const Matrix<float> &featMap, Matrix<uint16_t> *pLablVec)
{
  // determine the predicted class label from the output feature map
  int probVecLen = featMap.GetDimStp(0);
  Matrix<float> probLst(probVecLen);
  float *probVec = probLst.GetDataPtr();
  for (int dataInd = dataIndL; dataInd <= dataIndU; dataInd++)
  {
    // copy category probabilities to a temporary array
    memcpy(probVec, featMap.GetDataPtr(dataInd - dataIndL, 0, 0, 0),
           sizeof(float) * probVecLen);

    // determine the x-th predicted class label
    for (int lablInd = 0; lablInd < kLablCntPerData; lablInd++)
    {
      // find the maximal probability
      float probValOpt = FLT_MIN;
      uint16_t probValIndOpt = 0;
      for (int probValInd = 0; probValInd < probVecLen; probValInd++)
      {
        if (probValOpt < probVec[probValInd])
        {
          probValOpt = probVec[probValInd];
          probValIndOpt = probValInd;
        } // ENDIF: probValOpt
      }   // ENDFOR: probValInd

      // record current prediction
      probVec[probValIndOpt] = 0.0;
      pLablVec->SetEleAt(probValIndOpt, dataInd, lablInd, 0, 0);
    } // ENDFOR: lablInd
  }   // ENDFOR: dataInd
}

// INPUT REQUIREMENTS:
//   featMap: 1 x Cs x Hs x Ws
//   featBuf: (Ht x Wt) x (Ck x Hk x Wk)
void CaffeEva::CvtFeatMapToFeatBuf(
    const Matrix<float> &featMap, const int dataInd, const int grpInd,
    const LayerInfo &layerInfo, Matrix<float> *pFeatBuf)
{
  // obtain basic variables
  int imgChnSrc = featMap.GetDimLen(1);
  int imgHeiSrc = featMap.GetDimLen(2);
  int imgWidSrc = featMap.GetDimLen(3);
  int grpCnt = layerInfo.grpCnt;
  int padSiz = layerInfo.padSiz;
  int knlSiz = layerInfo.knlSiz;
  int stride = layerInfo.stride;
  int imgChnSrcPerGrp = imgChnSrc / grpCnt;
  int imgHeiDst = (imgHeiSrc + 2 * padSiz - (knlSiz - 1) - 1 + stride) / stride; //Fast ceiling
  int imgWidDst = ceil((imgWidSrc + 2 * padSiz - knlSiz) / stride) + 1;
  int rowCntBuf = pFeatBuf->GetDimLen(0);

  // copy feature data from <featMap> to <featBuf>
  int chnIndSrcL = grpInd * imgChnSrcPerGrp;
  memset(pFeatBuf->GetDataPtr(), 0, sizeof(float) * pFeatBuf->GetEleCnt());
  for (int rowIndBuf = 0; rowIndBuf < rowCntBuf; rowIndBuf++)
  {
    // obtain basic variables
    int widIndKnl = rowIndBuf % knlSiz;
    int heiIndKnl = rowIndBuf / knlSiz % knlSiz;
    int chnIndKnl = rowIndBuf / knlSiz / knlSiz;
    int heiIndDstL = std::max(0, (padSiz - heiIndKnl - 1 + stride) / stride); //Fast celing
    int heiIndDstU =
        std::min(imgHeiDst - 1, (padSiz - heiIndKnl + imgHeiSrc - 1) / stride);
    int heiCntDstSel = heiIndDstU - heiIndDstL + 1;
    int widIndDstL = std::max(0, (padSiz - widIndKnl - 1 + stride) / stride); // Use fast celing
    int widIndDstU =
        std::min(imgWidDst - 1, (padSiz - widIndKnl + imgWidSrc - 1) / stride);
    int widCntDstSel = widIndDstU - widIndDstL + 1;
    const float *ptrSrc =
        featMap.GetDataPtr(dataInd, chnIndSrcL + chnIndKnl, 0, 0);
    float *ptrDst = pFeatBuf->GetDataPtr(rowIndBuf, 0);

    // copy feature data from <featMap> to <featBuf>
    for (int heiIndDstSel = 0; heiIndDstSel < heiCntDstSel; heiIndDstSel++)
    {
      for (int widIndDstSel = 0; widIndDstSel < widCntDstSel; widIndDstSel++)
      {
        int heiIndDst = heiIndDstL + heiIndDstSel;
        int widIndDst = widIndDstL + widIndDstSel;
        int heiIndSrc = heiIndKnl + heiIndDst * stride - padSiz;
        int widIndSrc = widIndKnl + widIndDst * stride - padSiz;
        ptrDst[heiIndDst * imgWidDst + widIndDst] =
            ptrSrc[heiIndSrc * imgWidSrc + widIndSrc];
      } // ENDFOR: widIndDstSel
    }   // ENDFOR: heiIndDstSel
  }     // ENDFOR: rowIndBuf
}

// INPUT REQUIREMENTS:
//   featBuf: Ct x (Ht x Wt)
//   featMap: 1 x Ct x Ht x Wt
void CaffeEva::CvtFeatBufToFeatMap(
    const Matrix<float> &featBuf, const int dataInd, const int grpInd,
    const LayerInfo &layerInfo, Matrix<float> *pFeatMap)
{
  // obtain basic variables
  int imgChnDst = pFeatMap->GetDimLen(1);
  int chnIndDstL = imgChnDst / layerInfo.grpCnt * grpInd;

  // copy feature data from <featBuf> to <featMap>
  const float *ptrSrc = featBuf.GetDataPtr();
  float *ptrDst = pFeatMap->GetDataPtr(dataInd, chnIndDstL, 0, 0);
  memcpy(ptrDst, ptrSrc, sizeof(float) * featBuf.GetEleCnt());
}

void CaffeEva::GetDataMap(int indexL, int indexU, Matrix<float> *Output)
{
  CvtDataLstToFeatMap(indexL, indexU, dataLst, Output);
}

void CaffeEva::GetInPdMat(const Matrix<float> &dataLst,
                          const Matrix<float> &ctrdLst, Matrix<float> *pInPdMat)
{
  // obtain basic variables
  int dataCnt = dataLst.GetDimLen(0);
  int featDimCnt = dataLst.GetDimLen(1);
  int subSpaceCnt = ctrdLst.GetDimLen(0);
  int featCntPerSpace = ctrdLst.GetDimLen(1);
  int ctrdCntPerSpace = ctrdLst.GetDimLen(2);

  // resize the inner-product look-up table
  pInPdMat->Resize(dataCnt, subSpaceCnt, ctrdCntPerSpace);

  // compute the inner-product look-up table in each subspace
  for (int subSpaceInd = 0; subSpaceInd < subSpaceCnt; subSpaceInd++)
  {
    // determine the selected dimensions
    int featDimIndL = featCntPerSpace * subSpaceInd;
    int featDimCntSel = std::min(featDimCnt - featDimIndL, featCntPerSpace);

    // compute the inner-product look-up table for each instance
    const float *dataVec = dataLst.GetDataPtr(0, featDimIndL);
    float *inPdVec = pInPdMat->GetDataPtr(0, subSpaceInd, 0);
    for (int dataInd = 0; dataInd < dataCnt; dataInd++)
    {
      const float *ctrdVec = ctrdLst.GetDataPtr(subSpaceInd, 0, 0);
      memset(inPdVec, 0, sizeof(float) * ctrdCntPerSpace);
      for (int featDimInd = 0; featDimInd < featDimCntSel; featDimInd++)
      {
        cblas_saxpy(
            ctrdCntPerSpace, dataVec[featDimInd], ctrdVec, 1, inPdVec, 1);
        ctrdVec += ctrdCntPerSpace;
      } // ENDFOR: featDimInd

      // update pointers to the data vector and look-up table
      dataVec += featDimCnt;
      inPdVec += subSpaceCnt * ctrdCntPerSpace;
    } // ENDFOR: dataInd
  }   // ENDFOR: subSpaceInd
}

// Get the weightMap
// Not supposed to be called manually
// For the merging process
Matrix<float> *CaffeEva::GetWeightMatrix(int iLayer)
{

  LayerInfo &layerInfo = caffeParaObj.layerInfoLst[iLayer];
  LayerPara &layerPara = caffeParaObj.layerParaLst[iLayer];

  if (enblAprx)
  {
    if(layerInfo.type == ENUM_LyrType::PReLU)
      return &(layerPara.alphaVec);
    else
      return &(layerPara.ctrdLst); // Only return codebook
  }
  else
  {
    if (layerInfo.type == ENUM_LyrType::Conv)
    {
      return &(layerPara.convKnlLst);
    }
    if (layerInfo.type == ENUM_LyrType::FCnt)
    {
      return &(layerPara.fcntWeiMat);
    }
    if(layerInfo.type == ENUM_LyrType::PReLU)
    {
      return &(layerPara.alphaVec);
    }
  }
}

bool CaffeEva::VerifyTheBin(void)
{

  // display the greeting message
  printf("[CHECK-POINT] entering CaffeEva::VerifyTheBin()\n");

  // Declare a Original Model
  CaffePara OriginalModel;

  // Declare testing index CvtDataLstToFeatMap(dataIndL, dataIndU, dataLst, &(featMapLst[0]));
  int dataIndL = 0;
  int dataIndU = dataIndL + 0;

  // initialize <caffeParaObj>Quantized Model, original model
  caffeParaObj.Init(dirPathMain, fileNamePfx);
  OriginalModel.Init(dirPathMain, fileNamePfx);

  // load each layer's basic information
  if (modelName == "AlexNet")
  {
    caffeParaObj.ConfigLayer_AlexNet();
    OriginalModel.ConfigLayer_AlexNet();
  }
  else if (modelName == "CaffeNet")
  {
    caffeParaObj.ConfigLayer_CaffeNet();
    OriginalModel.ConfigLayer_CaffeNet();
  }
  else if (modelName == "VggCnnS")
  {
    caffeParaObj.ConfigLayer_VggCnnS();
    OriginalModel.ConfigLayer_VggCnnS();
  }
  else if (modelName == "VGG16")
  {
    caffeParaObj.ConfigLayer_VGG16();
    OriginalModel.ConfigLayer_VGG16();
  }
  else if (modelName == "CaffeNetFGB")
  {
    caffeParaObj.ConfigLayer_CaffeNetFGB();
    OriginalModel.ConfigLayer_CaffeNetFGB();
  }
  else if (modelName == "CaffeNetFGD")
  {
    caffeParaObj.ConfigLayer_CaffeNetFGD();
    OriginalModel.ConfigLayer_CaffeNetFGD();
  }
  else if (modelName == "VGG16Avg")
  {
    caffeParaObj.ConfigLayer_VGG16Avg();
    OriginalModel.ConfigLayer_VGG16Avg();
  }
  else if (modelName == "SoundCNN")
  {
    caffeParaObj.ConfigLayer_SoundCNN();
    OriginalModel.ConfigLayer_SoundCNN();
  }
  else if (modelName == "ZFNet")
  {
    caffeParaObj.ConfigLayer_ZFNet();
    OriginalModel.ConfigLayer_ZFNet();
  }
  else if (modelName == "LeNet")
  {
    caffeParaObj.ConfigLayer_LeNet();
    OriginalModel.ConfigLayer_LeNet();
  }
  else
  {
    printf("[ERROR] unrecognized caffe model name: %s\n", modelName.c_str());
    return false;
  } // ENDIF: modelName

  // load each layer's detailed parameters
  bool bcompressed = true;
  bool succFlg = caffeParaObj.LoadLayerPara(ENUM_AsmtEnc::Compact);
  bool succFlg2 = OriginalModel.LoadLayerPara(!bcompressed, ENUM_AsmtEnc::Compact);

  if (!succFlg)
  { // failed
    printf("[ERROR] Loading layer parameters");
    return false;
  } // ENDIF: succFlg

  if (!succFlg2)
  { // failed
    printf("[ERROR] Loading layer parameters");
    return false;
  } // ENDIF: succFlg

  PrepFeatMap();
  PrepFeatBuf();
  PrepCtrdBuf();
  PrepAsmtBuf();

  // Check each layer errors
  for (int layerInd = 0, IndexCONV = 0, IndexFC = 0; layerInd < caffeParaObj.layerCnt; layerInd++)
  {

    const LayerInfo &layerInfo = caffeParaObj.layerInfoLst[layerInd];
    const LayerPara &layerPara = caffeParaObj.layerParaLst[layerInd];
    const LayerPara &layerParaOriginal = OriginalModel.layerParaLst[layerInd];

    if (layerInfo.type == ENUM_LyrType::Conv)
    {
      IndexCONV++;
      printf("Check the convolutional layer #%d.\n", IndexCONV);

      Matrix<float> TempCONV(layerPara.convKnlLst);
      float *pfTempCONV = TempCONV.GetDataPtr();
      TempCONV.DispSizInfo();
      int iFilters = layerPara.asmtLst.GetDimLen(0);
      int iHeight = layerPara.asmtLst.GetDimLen(1);
      int iWidth = layerPara.asmtLst.GetDimLen(2);
      int subspaceCount = layerPara.asmtLst.GetDimLen(3);
      int CentroidCounts = layerPara.ctrdLst.GetDimLen(0);
      int ctrdCntPerSpace = layerPara.ctrdLst.GetDimLen(1);
      int CentroidSize = layerPara.ctrdLst.GetDimLen(2);
      int iTotal = TempCONV.GetEleCnt();

      assert(CentroidCounts * CentroidSize * iHeight * iWidth * iFilters == iTotal);

      // for(int indexFilter = 0 ; indexFilter < iFilters; indexFilter++){
      //   uint8_t *puAssign = layerPara.asmtLst.GetDataPtr(indexFilter, 0, 0, 0);

      //   for(int indexCentroid = 0; indexCentroid < CentroidCounts; indexCentroid ++ ) {

      //     float* pfCentroid = layerPara.ctrdLst.GetDataPtr(indexCentroid, 0, 0);
      //     memcpy(pfTempCONV, pfCentroid + puAssign[indexCentroid]*CentroidSize
      //           , sizeof(float) * CentroidSize );
      //     pfTempCONV += CentroidSize;
      //   }
      // }
    }
    if (layerInfo.type == ENUM_LyrType::FCnt)
    {
      IndexFC++;
      printf("Check the FC layer #%d. \n", IndexFC);

      Matrix<float> TempFC(layerPara.fcntWeiMat);
      float *pfTempFC = TempFC.GetDataPtr();
      //memcpy(pfTempFC, layerPara.fcntWeiMat.GetDataPtr(), sizeof(float)* TempFC.GetEleCnt());
      //TempFC.Permute(1, 0);
      TempFC.DispSizInfo();
      //TempFC.Resize(layerPara.fcntWeiMat.GetDimLen(1), layerPara.fcntWeiMat.GetDimLen(0));

      int iFilters = layerPara.asmtLst.GetDimLen(0);
      int subspaceCount = layerPara.asmtLst.GetDimLen(1);
      int CentroidCounts = layerPara.ctrdLst.GetDimLen(0);
      int CentroidSize = layerPara.ctrdLst.GetDimLen(2);
      int iTotal = TempFC.GetEleCnt();

      if (IndexFC == 1)
      {

        for (size_t i = 0; i < 4096; i++)
        {
          printf("%.8f ", pfTempFC[i]);
        }
        printf("\n");
      }
      assert(CentroidCounts * CentroidSize * iFilters == iTotal);

      for (int indexFilter = 0; indexFilter < iFilters; indexFilter++)
      {
        uint8_t *puAssign = layerPara.asmtLst.GetDataPtr(indexFilter, 0);

        for (int indexCentroid = 0; indexCentroid < CentroidCounts; indexCentroid++)
        {

          float *pfCentroid = layerPara.ctrdLst.GetDataPtr(indexCentroid, 0, 0);
          memcpy(pfTempFC, pfCentroid + puAssign[indexCentroid] * CentroidSize, sizeof(float) * CentroidSize);
          pfTempFC += CentroidSize;
        }
      }

      // Permute the First FC layer if needed
      if (layerInfo.arrang == ENUM_LyrArrangement::HeightWidthChannel)
      {

        Matrix<float> *FeatMap = this->GetFeatMap(layerInd);
        int iHeight = FeatMap->GetDimLen(1);
        int iWidth = FeatMap->GetDimLen(2);
        int iInputChannel = FeatMap->GetDimLen(3);

        TempFC.Resize(iFilters, iHeight, iWidth, iInputChannel);
        TempFC.Permute(0, 3, 1, 2); //Convert to Channel Height Width
        TempFC.Resize(iFilters, CentroidSize * CentroidCounts);
      }

      double difference = 0;

      pfTempFC = TempFC.GetDataPtr();
      float *pfOriginal = layerParaOriginal.fcntWeiMat.GetDataPtr();
      for (int i = 0; i < iTotal; i++)
      {
        double error = fabs(pfTempFC[i] - pfOriginal[i]);
        difference = difference + error;
        if (error > 100)
          printf("Something wrong! at index %d", i);
      }
      printf("Sum of error at #%d FC = %7.4f, average = %7.4f\n",
             IndexFC, difference, difference / iTotal);
    }

  } // ENDFOR: layerInd

  printf("End of check bin file.\n");
}
