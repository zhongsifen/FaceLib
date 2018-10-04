//
//  FaceLib.hpp
//  FaceLib
//
//  Created by SIFEN ZHONG on 1/10/2018.
//  Copyright © 2018 ___ZHONGSIFEN___. All rights reserved.
//

#ifndef FaceLib_h
#define FaceLib_h

#include "face_detection.h"
#include "face_alignment.h"
#include "face_identification.h"
#include <opencv2/opencv.hpp>

class FaceLib {
public:
	seeta::FaceDetection* det;
	seeta::FaceAlignment* ali;
	seeta::FaceIdentification* net;
	
public:
	FaceLib() { setup(); }
	bool setup();
	
};

void saveFeaturesFilePair(std::pair<std::vector<std::string>, std::vector<std::vector<float> >>  &features, std::string &filename);

void loadFeaturesFilePair(std::pair<std::vector<std::string>, std::vector<std::vector<float> >> &features, std::string &filename);

bool extractFeat(seeta::FaceDetection &detector, seeta::FaceAlignment &point_detector, seeta::FaceIdentification &face_recognizer, cv::Mat &img_color, cv::Mat dst_img, float * feat);

#endif /* FaceLib_h */
