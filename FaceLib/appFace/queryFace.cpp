
#include "FaceLib/FaceLib.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>

#include <sstream>
#include <fstream>
#include <glob.h>

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

using namespace seeta;
using namespace std;

const std::string _M("/Users/zhongsifen/Work/SeetaFaceModel/");
const std::string _D("/Users/zhongsifen/Work/SeetaFaceData/");

int main_query(int argc, char* argv[]) {
    
    // Initialize face detection model
    seeta::FaceDetection detector(_M + "seeta_fd_frontal_v1.0.bin");
    detector.SetMinFaceSize(20);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);
    
    std::pair<vector<string>, vector<vector<float> >>  namesFeats;
    
    // Initialize face alignment model and face Identification model
    seeta::FaceAlignment point_detector(_M + "seeta_fa_v1.1.bin");
    FaceIdentification face_recognizer(_M + "seeta_fr_v1.0.bin");
    
    cv::Mat dst_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
    
    // Save Cropped faces
    string path_imgCroppedNames = _D + "crop";
    
    // Load image names and features
    string filenamePair = "namesFeats.bin";
    loadFeaturesFilePair(namesFeats, filenamePair);

    // Extract query face image feature
    float queryFeat[2048];
    cv::Mat queryImg_color = cv::imread(_D + "test_face_recognizer/images/src/NF_200001_001.jpg");
    extractFeat(detector, point_detector, face_recognizer, queryImg_color, dst_img, queryFeat);
    
    // Calculate cosine distance between query and data base faces
    vector<pair<float,size_t> > dists_idxs;
    int i = 0;
    for(auto featItem: namesFeats.second){
        // http://stackoverflow.com/questions/2923272/how-to-convert-vector-to-array-c
        float tmp_cosine_dist = face_recognizer.CalcSimilarity(&namesFeats.second[0][0], &featItem[0]);
        dists_idxs.push_back(std::make_pair(tmp_cosine_dist, i++));
    }
    
    // Sorting will put lower values ahead of larger ones, resolving ties using the original index
    std::sort(dists_idxs.begin(), dists_idxs.end());
    std::reverse(dists_idxs.begin(), dists_idxs.end());
    for (size_t i = 0 ; i != dists_idxs.size() ; i++) {
        printf("distance: %f, face image: %s\n", dists_idxs[i].first, namesFeats.first.at(dists_idxs[i].second).c_str());
    }
    
    return 0;
}
