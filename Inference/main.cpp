/*==================================================================================================== 
 MIT License

 Copyright (c) 2018 Yi-Ming Chan, yimingchan@gmail.com 
  Image & Vision Computing Lab, Institute of Information Science, Academia Sinica 

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ====================================================================================================*/

#include "merged-cnn/mergedCaffeEva.h"
#include <string.h>
#include <sstream> // For extract words
#include <fstream> // Copy file
#include <cstdio>
#include "quantized-cnn/include/FileIO.h"

using std::string;
using std::vector;

// Extract list function
 bool ExtractLists(const std::string& filePfxSrc, std::vector< std::string> &Lists)
 {
  // List for the merged model
  FILE* fList;
  fList = fopen(filePfxSrc.c_str(), "r");
  if (fList == nullptr) {
    printf("[ERROR] could not open file %s\n", filePfxSrc.c_str());
	printf("[DEBUG] File string length= %d\n", filePfxSrc.length());

    return false;
  }  
  
  char textLine[256];
  while( fgets(textLine, sizeof(textLine), fList) ){
    //char *ModelName = strtok(textLine, " ");
	if(textLine[0]=='\n')
		continue;
    std::string sList(textLine);
    Lists.push_back(sList);
    //printf("Model %s\n", sModels.c_str());
  }
  fclose(fList);
  
  return true;
 }
 
 // Write QCNN compaitable bin file
 bool WriteQCNNFile(const string Name, const LayerInfo &layerInfo, const CaffePara& Model, int iLayerIndex)
 {
 	const int kStrBufLen = 256;
  char strBuf[kStrBufLen];
  bool succFlg = true;	
	std::ifstream ReadBinFile(Name.c_str(), std::fstream::binary);
		
	if(ReadBinFile.fail())
	{
		printf("Read file failed. %s\n",Name.c_str());
		return false;
	}
	if (Name.find(".weight") != string::npos)
	{
		printf("Writing weight file.\n");
		if(layerInfo.type == ENUM_LyrType::Conv)
		{
		   snprintf(strBuf, kStrBufLen, "%s/%s.convKnl.%02d.bin",
                 Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
		}
		if(layerInfo.type == ENUM_LyrType::FCnt)
		{
		   snprintf(strBuf, kStrBufLen, "%s/%s.fcntWei.%02d.bin",
                 Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
		}
		if(layerInfo.type == ENUM_LyrType::PReLU)
		{
		   snprintf(strBuf, kStrBufLen, "%s/%s.preluWei.%02d.bin",
                 Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
		}
	}
	if(Name.find("index") != string::npos)
	{
       snprintf(strBuf, kStrBufLen, "%s/%s.asmtLst.%02d.bin",
                   Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
	}
	if(Name.find("codebook") != string::npos)
	{
       snprintf(strBuf, kStrBufLen, "%s/%s.ctrdLst.%02d.bin",
                 Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
	}
	if(Name.find(".bias") != string::npos)
	{
		printf("Writing bias file. \n");
		snprintf(strBuf, kStrBufLen, "%s/%s.biasVec.%02d.bin",
                 Model.dirPath.c_str(), Model.filePfx.c_str(), iLayerIndex + 1);
	}
	
	// Write to the file
	std::ofstream WriteBinFile(strBuf, std::fstream::trunc|std::fstream::binary);
	WriteBinFile << ReadBinFile.rdbuf();

	return true;
}
bool ConvertBinFileList(const std::string sModelFile, const CaffePara& Model)
{
	std::string Path;

	
	Path = sModelFile.substr(0,sModelFile.find_last_of("/\\")+1);

	printf("File path = %s \n", Path.c_str());

	vector<string> FileList;

	if(!ExtractLists(sModelFile, FileList))
		printf("[ERROR] Read Model File Failed!\n");
	else
		printf("[INFO] Read Model File: %s\n", sModelFile.c_str());

	int iLayers = Model.layerCnt;
	int idxList = 0;
	std::stringstream ssFileName;
	for(int i=0; i < iLayers; i++)
	{
	  const LayerInfo &layerInfo = Model.layerInfoLst[i];
      //LayerPara &layerPara = Model.layerParaLst[i];
	  if( (layerInfo.type == ENUM_LyrType::Conv) ||
          (layerInfo.type == ENUM_LyrType::FCnt)  )
	  {
			//printf("Layer type = %d\n",layerInfo.type);
			// Print weight and bias name
					string Name;
			ssFileName << FileList[idxList++]; ssFileName >> Name; 
			if(Name.find(".bin") != string::npos)
				Name = Path + Name;
			else
				Name = Path + Name + ".bin";
			printf("%s \n", Name.c_str());
			WriteQCNNFile(Name, layerInfo, Model, i);
		
			// Read next line if it is not the end
			if(i == iLayers - 1)
				continue;
			// Check if there is a bias file in the list
			ssFileName << FileList[idxList++]; ssFileName >> Name; 
			if(Name.find("bias") == string::npos)
			{
				idxList--; continue; // This list dose not contain bias
			}
			else // Output bias file
			{
				/* code */
				if(Name.find(".bin") != string::npos)
					Name = Path + Name;
				else
					Name = Path + Name + ".bin";
				printf("%s \n", Name.c_str());
				WriteQCNNFile(Name, layerInfo, Model, i);
			}
		
	  }
	  if (layerInfo.type == ENUM_LyrType::PReLU) 
	  {
			// Print PReLU name
			//printf("Layer type = %d\n",layerInfo.type);
			ssFileName << FileList[idxList++];
			string Name;
			ssFileName >> Name; 
			if(Name.find("relu") == string::npos)
			{
					idxList--; continue; // This list dose not contain PReLU
			}
			else
			{
				/* code */
				if(Name.find(".bin") != string::npos)
					Name = Path + Name;
				else
					Name = Path + Name + ".bin";
				printf("%s \n", Name.c_str());
				WriteQCNNFile(Name, layerInfo, Model, i);
			}
	  }
	}
}

// Convert the file into QCNN style
int ConvertFile(int argc, char* argv[]) 
{
    //
    std::vector< std::string> Models;
    bool succ;
    if (argc > 1)
      succ = ExtractLists(argv[1], Models);
    else
      succ = ExtractLists("./testList.txt", Models);
    if(!succ)
      printf("Something wrong!\n");
    else{
      int iNumberOfModels = Models.size();
      for(size_t i = 0; i < iNumberOfModels; i++)
      {
        std::stringstream ss;//(Models[i]);
        ss << Models[i];
        std::string item;
        std::vector< std::string> sConfigs;
        while(ss >> item){
          sConfigs.push_back(item);
        }
		    CaffePara caffeParaObj;
        // load parameters for the caffe model
        caffeParaObj.Init("./quantized-cnn/Models/merged_bin_file", "SphereFace20");
				 //caffeParaObj.Init("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
        caffeParaObj.ConfigLayer_SphereFace20();
		    ConvertBinFileList(sConfigs[0],caffeParaObj); // Convert to QCNN style model file
		
				//std::string ModelFile("./quantized-cnn/Models/SphereFace20_sp/list.txt");
				//caffeParaObj.LoadLayerPara(false, ENUM_AsmtEnc::Raw, ModelFile);
				//caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Raw, ModelFile);
   
      }
   
    }
   
 
  return 0;
}

void ForwardTest()
{
	// Important, Convert index into compact first
	CaffePara caffeParaObj;
	caffeParaObj.Init("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
	caffeParaObj.ConfigLayer_SphereFace20();
	caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Raw);
        caffeParaObj.CvtAsmtEnc(ENUM_AsmtEnc::Raw, ENUM_AsmtEnc::Compact);
        caffeParaObj.LoadLayerPara(true, ENUM_AsmtEnc::Compact);

	MergedCaffeEva Caffe1, Caffe2;
	StopWatch stopWatchObj; // For timer
	float Caffe1Time, Caffe2Time;

	std::string kCaffeModelName1 = "SphereFace20Original";
	std::string kCaffeModelName2 = "SphereFace20PQ";
	// Model parameter choose
	bool EnblAprxComp = true;
	// Init, always use quantized version
	stopWatchObj.Reset();
	stopWatchObj.Resume();
	
	for(int i=0; i < 4; i++)
	{	
	Caffe1.Init(!EnblAprxComp);
	Caffe1.SetModelName("SphereFace20");
	Caffe1.SetModelPath("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
	Caffe1.LoadDatasetonly("./quantized-cnn/Data/VoxForge.112x112.IMG");
	Caffe1.LoadCaffePara();
	Caffe1.ExecForwardPass();
	}
	stopWatchObj.Pause();
	//printf("elapsed time of Model %s: %.4f (s)\n", kCaffeModelName1.c_str(), stopWatchObj.GetTime());
	double CaffeTime1 = stopWatchObj.GetTime();
	Caffe1Time = Caffe1.DispElpsTime();
	
	stopWatchObj.Reset();
	stopWatchObj.Resume();
	
	for(int i=0; i < 4; i++)
	{	
	Caffe2.Init(EnblAprxComp);
	Caffe2.SetModelName("SphereFace20");
	Caffe2.SetModelPath("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
	Caffe2.LoadDatasetonly("./quantized-cnn/Data/VoxForge.112x112.IMG");
	Caffe2.LoadCaffePara();
	Caffe2.ExecForwardPass();
	}
	stopWatchObj.Pause();
	//printf("elapsed time of Model %s: %.4f (s)\n", kCaffeModelName2.c_str(), stopWatchObj.GetTime());
	double CaffeTime2 = stopWatchObj.GetTime();	
	Caffe2Time = Caffe2.DispElpsTime();

	printf("Overall Speedup = %.4f (%.4f, %.4f)\n", Caffe1Time/Caffe2Time, CaffeTime1, CaffeTime2);

    // Verify the output
	// Matrix<float > outputLst;
	// std::string strBuf = "./quantized-cnn/Data/VoxForge.112x112.IMG/100pic_predict_emb.bin";
	// //std::string strBuf = "./quantized-cnn/Data/VoxForge.112x112.IMG/1pic_predict_emb.bin";
	// //std::string strBuf = "./quantized-cnn/Data/VoxForge.112x112.IMG/1pic_predict_afterfirstprelu.bin";
	// //std::string strBuf = "./quantized-cnn/Data/VoxForge.112x112.IMG/1pic_predict_beforeflatten.bin";
	// //std::string strBuf = "./quantized-cnn/Data/VoxForge.112x112.IMG/1pic_predict_firstshortcut.bin";
	// Caffe1.LoadOutputonly(strBuf);
	// Caffe1.EvaluateOutput(49);

}

void MergeTest()
{

	MergedCaffeEva Caffe1, Caffe2;
	StopWatch stopWatchObj; // For timer

	std::string kCaffeModelName1 = "SphereFace20Original";
	std::string kCaffeModelName2 = "SphereFace20PQ";
	// Model parameter choose
	bool EnblAprxComp = true;
	// Init, always use quantized version
	Caffe1.Init(EnblAprxComp);
	Caffe1.SetModelName("SphereFace20");
	Caffe1.SetModelPath("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
	Caffe1.LoadDatasetonly("./quantized-cnn/Data/VoxForge.112x112.IMG");
	Caffe1.LoadCaffePara();
	
	Caffe2.Init(EnblAprxComp);
	Caffe2.SetModelName("SphereFace20");
	Caffe2.SetModelPath("./quantized-cnn/Models/SphereFace20_sp", "SphereFace20");
	Caffe2.LoadDatasetonly("./quantized-cnn/Data/VoxForge.112x112.IMG");
	Caffe2.LoadCaffePara();
	
	
	std::vector<int> index;
	index = {0,2, 4, 7, 9, 11, 14, 16, 19, 21, 23, 26, 28, 31, 33, 36, 38, 41, 43, 45, 48};
	
	//Caffe1.MergeModel(Caffe1, Caffe2);
	//Caffe1.MergeModel(Caffe2);
	//Caffe1.MergeModel(Caffe2, Caffe1, index);
	Caffe1.MergeModel(Caffe2, index);

	Caffe1.ExecForwardPass();
	Caffe2.ExecForwardPass();

	printf("Success!\n");

	
}


int main(int argc, char* argv[])
{
	//openblas_set_num_threads(4);	
	// Convert BinFile into QCNN style
	//ConvertFile(argc, argv);
	
	// Test forward with time;
	ForwardTest();

	// Test of model combined
	//MergeTest();
	
	return 0;
}
