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

#ifndef MERGEDCAFFEEVA_H
#define MERGEDCAFFEEVA_H

#include "../quantized-cnn/include/CaffeEva.h"

class MergedCaffeEva : public CaffeEva
{
  
  public: 
    // Merge codebooks in the model
    bool MergeModel( MergedCaffeEva& model);
    // Merge codebooks in the model with different size
    bool MergeModel( MergedCaffeEva& model, std::vector<int> & mapping);
    // Merge codebooks in two Models
    bool MergeModel( MergedCaffeEva& m1, MergedCaffeEva& m2);
    // Merge codebooks in two Models with different size
    bool MergeModel( MergedCaffeEva& m1, MergedCaffeEva& m2, std::vector< int>& mapping);
    // Assign the codebook to the i-th layer
    bool MergeCodeBook(int LayerIndex, Matrix<float> &Codebook);

  private:

};

#endif
