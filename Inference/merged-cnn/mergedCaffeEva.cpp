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

#include "mergedCaffeEva.h"

bool MergedCaffeEva::MergeCodeBook(int iLayerIndex, Matrix<float> &Codebook)
{
    Matrix<float> *pWeight = this->GetWeightMatrix(iLayerIndex);
    if(pWeight->GetDimLen(0) == Codebook.GetDimLen(0) &&
       pWeight->GetDimLen(1) == Codebook.GetDimLen(1) &&
       pWeight->GetDimLen(2) == Codebook.GetDimLen(2)      )
    {
        // Merge two codebook
        *pWeight = Codebook;
    }
    else {
        printf("ERROR! Cannot merge two codebooks!\n");
        return 0;
    }
    return true;
}


bool MergedCaffeEva::MergeModel(MergedCaffeEva& m1, MergedCaffeEva& m2)
{
    int iLayerCount1 = m1.GetLayerCount();    
    int iLayerCount2 = m2.GetLayerCount();
    Matrix<float> * model1Weight;
    if(!m1.IsApproximate() || !m2.IsApproximate())
    {
        printf("No need to merge precise models.\n");
        return false;
    }
    if(iLayerCount1 != iLayerCount2)
    {
        printf("Error to merge differe size model without mapping!\n");
        return false;
    }
    else
    {
        for(size_t i = 0; i < iLayerCount1; i++)
        {
            /* code */
            model1Weight = m1.GetWeightMatrix(i);
            // Assume only codebook will be used
            // Codebook is assumed to be dim = 3
            if(model1Weight->GetEleCnt() > 0 && model1Weight->GetDimCnt() > 1)
            {
                m2.MergeCodeBook(i, *model1Weight);
            }
        }
        
    }
    return true;
    
}

// Absorbing one model
bool MergedCaffeEva::MergeModel(MergedCaffeEva& model)
{
    return this->MergeModel(*this, model);
}

// mapping vector is required for merging models with different size
// the mapping vector contain the index of the merged layer
bool MergedCaffeEva::MergeModel( MergedCaffeEva& m1, MergedCaffeEva& m2, std::vector< int>& mapping)
{
    int iLayerCount1 = m1.GetLayerCount();
    int iLayerCount2 = m2.GetLayerCount();

    Matrix<float> *modelWeight;
    Matrix<float> *modelWeightTemp;
    if(!m1.IsApproximate() || !m2.IsApproximate())
    {
        printf("No need to merge precise models.\n");
        return false;
    }
    int iMaxCount, iMinCount;
    MergedCaffeEva *maxModel, *minModel;

    if(iLayerCount1 >= iLayerCount2)
    {
        iMaxCount = iLayerCount1; 
        iMinCount = iLayerCount2;
        maxModel = &m1; 
        minModel = &m2;
    }
    else
    {
        iMaxCount = iLayerCount2;
        iMinCount = iLayerCount1;
        maxModel = &m2;
        minModel = &m1;
    }
    
    int idxMapping = 0;
    for(size_t i = 0, j = 0; i < iMaxCount; i++)
    {
        if( i != mapping[idxMapping])
        {
            continue;
        }
        else
        {
            while( j < iMinCount)
            {
                modelWeightTemp = minModel->GetWeightMatrix(j);
                // Check if this is model is 
                if(modelWeightTemp->GetEleCnt() == 0 ||
                   modelWeightTemp->GetDimCnt() < 3)
                {
                    j++; //Skip, go to next layer
                }
                else
                {
                    break;
                }
            }

            modelWeight = maxModel->GetWeightMatrix(i);
            
            minModel->MergeCodeBook(j, *modelWeight);
            j++;
            idxMapping++;
        }

    }
    
    return true;
}

// mapping vector is required for merging models with different size
// the mapping vector contain the index of the merged layer
bool MergedCaffeEva::MergeModel( MergedCaffeEva& model, std::vector< int>& mapping)
{
    return this->MergeModel(*this, model, mapping);
}