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
}