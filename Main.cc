/*
 * Copyright © CASIA 2015-2016.
 *
 * Paper: Quantized Convolutional Neural Networks for Mobile Devices (CVPR 2016)
 * Authors: Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, and Jian Cheng
 */

// Modified by : Yi-Ming Chan

#include "include/UnitTest.h"

int main(int argc, char* argv[]) {
  // run unit-test for the <CaffePara> class
  //UnitTest::UT_CaffePara();
  UnitTest::UT_Models(); // Conver the bin file
  // MODE 1: SPEED TEST
  // run unit-test for the <CaffeEva> class
  //UnitTest::UT_CaffeEva();
  UnitTest::UT_Tensorflow(1);

  // MODE 2: SINGLE IMAGE CLASSIFICATION
  // run unit-test for the <CaffeEvaWrapper> class
  // UnitTest::UT_CaffeEvaWrapper();


  return 0;
}
