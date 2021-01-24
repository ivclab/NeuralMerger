/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

#include "../include/UnitTest.h"

#include "../include/CaffeEva.h"
#include "../include/CaffeEvaWrapper.h"
#include "../include/CaffePara.h"
#include "../include/StopWatch.h"
//#define _DEBUG

void UnitTest::UT_CaffePara(void) {
  // create class objects for unit-test
  CaffePara caffeParaObj;

  // load parameters for the caffe model
  caffeParaObj.Init("./AlexNet/Bin.Files", "bvlc_alexnet_aCaF");
  caffeParaObj.ConfigLayer_AlexNet();
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Compact);
}

void UnitTest::UT_Models(void) {
  // create class objects for unit-test
  CaffePara caffeParaObj, caffeParaObj2, caffeParaObj3, caffeParaObj4;

  // load parameters for the soundCNN model
  caffeParaObj.Init("./Models/SoundCNN/Bin.Files", "soundCNN_aCaF");
  caffeParaObj.ConfigLayer_SoundCNN();
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Compact);

  caffeParaObj2.Init("./Models/LeNet/Bin.Files", "LeNet_aCaF");
  caffeParaObj2.ConfigLayer_LeNet();
  caffeParaObj2.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj2.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj2.LoadLayerPara(true, ENUM_AsmtEnc::Compact);

  caffeParaObj3.Init("./Models/VGG16", "vgg16Avg_aCaF");
  caffeParaObj3.ConfigLayer_VGG16Avg();
  caffeParaObj3.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj3.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj3.LoadLayerPara(true, ENUM_AsmtEnc::Compact);

  caffeParaObj4.Init("./Models/ZFNet", "ZF_aCaF");
  caffeParaObj4.ConfigLayer_ZFNet();
  caffeParaObj4.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
  caffeParaObj4.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
  caffeParaObj4.LoadLayerPara(true, ENUM_AsmtEnc::Compact);

}

void UnitTest::UT_CaffeEva(void) {
  // create class objects for unit-test
  StopWatch stopWatchObj;
  CaffeEva caffeEvaObj;

  // choose a caffe model
  bool kEnblAprxComp = true;
  const std::string kCaffeModelName = "AlexNet";

  // evaluate the caffe model's classification accuracy
  stopWatchObj.Reset();
  stopWatchObj.Resume();
  caffeEvaObj.Init(kEnblAprxComp);  // enable approximate but fast computation
  if (kCaffeModelName == "AlexNet") {
    caffeEvaObj.SetModelName("AlexNet");
    caffeEvaObj.SetModelPath("./AlexNet/Bin.Files", "bvlc_alexnet_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.227x227.IMG");
  } else if (kCaffeModelName == "CaffeNet") {
    caffeEvaObj.SetModelName("CaffeNet");
    caffeEvaObj.SetModelPath("./CaffeNet/Bin.Files", "bvlc_caffenet_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.227x227.IMG");
  } else if (kCaffeModelName == "VggCnnS") {
    caffeEvaObj.SetModelName("VggCnnS");
    caffeEvaObj.SetModelPath("./VggCnnS/Bin.Files", "vgg_cnn_s_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.224x224.IMG");
  } else if (kCaffeModelName == "VGG16") {
    caffeEvaObj.SetModelName("VGG16");
    caffeEvaObj.SetModelPath("./VGG16/Bin.Files", "vgg16_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.224x224.PXL");
  } else if (kCaffeModelName == "VGG16Avg") {
    caffeEvaObj.SetModelName("VGG16Avg");
    caffeEvaObj.SetModelPath("./VGG16Avg/Bin.Files", "vgg16Avg_aCaF");
    caffeEvaObj.LoadDataset("./ILSVRC12.224x224.IMG");
  } else if (kCaffeModelName == "SoundCNN") {
    caffeEvaObj.SetModelName("SoundCNN");
    caffeEvaObj.SetModelPath("./SoundCNN/Bin.Files", "soundCNN_aCaF");
    caffeEvaObj.LoadDataset("./MNIST.32x32.IMG");
  } else if (kCaffeModelName == "LeNet") {
    caffeEvaObj.SetModelName("LeNet");
    caffeEvaObj.SetModelPath("./LeNet/Bin.Files", "LeNet_aCaF");
    caffeEvaObj.LoadDataset("./MNIST.32x32.IMG");    
  }else {
    printf("[ERROR] unrecognized caffe model: %s\n", kCaffeModelName.c_str());
  }  // ENDIF: kCaffeModelName
  caffeEvaObj.LoadCaffePara();
  caffeEvaObj.ExecForwardPass();
  caffeEvaObj.CalcPredAccu();
  caffeEvaObj.DispElpsTime();
  stopWatchObj.Pause();
  printf("elapsed time: %.4f (s)\n", stopWatchObj.GetTime());
}

void UnitTest::UT_Tensorflow(int experiments) {
  // create class objects for unit-test
  StopWatch stopWatchObj;
  CaffeEva caffeEvaObj, caffeEvaObj2;
  
  // choose a caffe model
  bool kEnblAprxComp = true;
  bool isFashion = true;
  std::string kCaffeModelName = "SoundCNN";
  std::string kCaffeModelName2 = "LeNet";
  
  int iIterations = 1;
  int iTestCount = 1;
  int layers;

  double dTotalTime1, dTotalTime2;
  
  switch( experiments )
  {
    case 1:
      // Model 1
      stopWatchObj.Reset();
      stopWatchObj.Resume();
      caffeEvaObj2.Init(kEnblAprxComp);
      caffeEvaObj2.SetModelName("LeNet");
      caffeEvaObj2.SetModelPath("./Models/LeNet/Bin.Files", "LeNet_aCaF");
      if (isFashion)
        caffeEvaObj2.LoadDataset("./Data/FASHION.32x32.IMG");  
      else
        caffeEvaObj2.LoadDataset("./Data/MNIST.32x32.IMG");  

      caffeEvaObj2.LoadCaffePara();
      for(int i=0; i < iIterations; i++) {
        caffeEvaObj2.ExecForwardPass();
      }
      stopWatchObj.Pause();
      dTotalTime2 = caffeEvaObj2.DispElpsTime();
      printf("elapsed time of Model %s: %.4f (s)(from watch) %.4f (s)(by object) \n", kCaffeModelName2.c_str(), stopWatchObj.GetTime(), dTotalTime2);

      // evaluate the caffe model's classification accuracy
      stopWatchObj.Reset();
      stopWatchObj.Resume();
      caffeEvaObj.Init(kEnblAprxComp);  // enable approximate but fast computation
      if (kCaffeModelName == "AlexNet") {
        caffeEvaObj.SetModelName("AlexNet");
        caffeEvaObj.SetModelPath("./AlexNet/Bin.Files", "bvlc_alexnet_aCaF");
        caffeEvaObj.LoadDataset("./Data/ILSVRC12.227x227.IMG");
      } else if (kCaffeModelName == "CaffeNet") {
        caffeEvaObj.SetModelName("CaffeNet");
        caffeEvaObj.SetModelPath("./CaffeNet/Bin.Files", "bvlc_caffenet_aCaF");
        caffeEvaObj.LoadDataset("./Data/ILSVRC12.227x227.IMG");
      } else if (kCaffeModelName == "VggCnnS") {
        caffeEvaObj.SetModelName("VggCnnS");
        caffeEvaObj.SetModelPath("./VggCnnS/Bin.Files", "vgg_cnn_s_aCaF");
        caffeEvaObj.LoadDataset("./ILSVRC12.224x224.IMG");
      } else if (kCaffeModelName == "VGG16") {
        caffeEvaObj.SetModelName("VGG16");
        caffeEvaObj.SetModelPath("./VGG16/Bin.Files", "vgg16_aCaF");
        caffeEvaObj.LoadDataset("./ILSVRC12.224x224.PXL");
      } else if (kCaffeModelName == "VGG16Avg") {
        caffeEvaObj.SetModelName("VGG16Avg");
        caffeEvaObj.SetModelPath("./VGG16Avg/Bin.Files", "vgg16Avg_aCaF");
        caffeEvaObj.LoadDataset("./ILSVRC12.224x224.IMG");
      } else if (kCaffeModelName == "SoundCNN") {
        caffeEvaObj.SetModelName("SoundCNN");
        caffeEvaObj.SetModelPath("./Models/SoundCNN/Bin.Files", "soundCNN_aCaF");
        caffeEvaObj.LoadDataset("./Data/Sound.32x32.IMG");
      } else if (kCaffeModelName == "LeNet") {
        caffeEvaObj.SetModelName("LeNet");
        caffeEvaObj.SetModelPath("./Models/LeNet/Bin.Files", "LeNet_aCaF");
        if (isFashion)
          caffeEvaObj.LoadDataset("./Data/FASHION.32x32.IMG");  
        else
          caffeEvaObj.LoadDataset("./Data/MNIST.32x32.IMG"); 
          
      }else {
        printf("[ERROR] unrecognized caffe model: %s\n", kCaffeModelName.c_str());
      }  // ENDIF: kCaffeModelName
      caffeEvaObj.LoadCaffePara();
      for(int i=0; i < iIterations; i++) {
        caffeEvaObj.ExecForwardPass();
      }
      stopWatchObj.Pause();
      dTotalTime1 = caffeEvaObj.DispElpsTime();
      printf("elapsed time of Model %s: %.4f (s)(from watch) %.4f (s)(by obj)\n", kCaffeModelName.c_str(), stopWatchObj.GetTime(), dTotalTime1); 
      
      printf("Model %s:\n", kCaffeModelName.c_str());
      caffeEvaObj.CalcPredAccu();
      printf("Model %s:\n", kCaffeModelName2.c_str());
      caffeEvaObj2.CalcPredAccu();
      break;

    case 2:
      // Model 2
      stopWatchObj.Reset();
      stopWatchObj.Resume();
      caffeEvaObj2.Init(kEnblAprxComp);
      caffeEvaObj2.SetModelName("VGG16Avg");
      caffeEvaObj2.SetModelPath("./Models/VGG16", "vgg16Avg_aCaF");
      caffeEvaObj2.LoadDataset("./Data/MVC.224x224.IMG");  
      kCaffeModelName2 = "VGG16Avg"; 
    
      caffeEvaObj2.LoadCaffePara();
      layers = caffeEvaObj2.GetLayerCount();
      for(int i=0; i < iIterations; i++) {
        caffeEvaObj2.ExecForwardPass();
      }
      stopWatchObj.Pause();
      caffeEvaObj2.DispElpsTime();
      printf("elapsed time of Model %s: %.4f (s)\n", kCaffeModelName2.c_str(), stopWatchObj.GetTime()); 

      // evaluate the caffe model's classification accuracy
      stopWatchObj.Reset();
      stopWatchObj.Resume();

      // Model 1
      caffeEvaObj.Init(kEnblAprxComp);
      caffeEvaObj.SetModelName("ZFNet");
      caffeEvaObj.SetModelPath("./Models/ZFNet", "ZF_aCaF");
      caffeEvaObj.LoadDataset("./Data/Gender.227x227.IMG");  
      kCaffeModelName = "ZFNet";

      caffeEvaObj.LoadCaffePara();
      for(int i=0; i < iIterations; i++) {       
        caffeEvaObj.ExecForwardPass();
      }

      stopWatchObj.Pause();
      caffeEvaObj.DispElpsTime();
      printf("elapsed time of Model %s: %.4f (s)\n", kCaffeModelName.c_str(), stopWatchObj.GetTime());

      // evaluate the caffe model's classification accuracy
      printf("Model %s:\n", kCaffeModelName.c_str());
      caffeEvaObj.CalcPredAccu();
      printf("Model %s:\n", kCaffeModelName2.c_str());
      caffeEvaObj2.CalcPredAccu();
      break;

      break;
    
    default:
      
      break;
  }

}


