#ifndef ELLIPSOIDSLAM_FRAME_H
#define ELLIPSOIDSLAM_FRAME_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Ellipsoid.h"

#include <src/Relationship/Relationship.h>

namespace EllipsoidSLAM{
class Frame;
class Observation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int label;
    Vector4d bbox;   // left-top x1 y1, right-down x2 y2
    double rate;    // accuracy:   0 - 1.0
    Frame* pFrame;  // which frame is the observation from

    int instance;  // useless , for debugging

    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descripts;
    cv::Mat gray;

    cv::Mat dtMat;
    cv::Mat edgesMat;
};
typedef std::vector<Observation*> Observations;

class Observation3D {
public:
    int label;
    g2o::ellipsoid* pObj;
    double rate;    // prob:   0 - 1.0
    Frame* pFrame;  
};
typedef std::vector<Observation3D*> Observation3Ds;

class Measurement
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int measure_id;
    int instance_id;    // associated object instance id
    Observation ob_2d;
    Observation3D ob_3d;
};
typedef std::vector<Measurement> Measurements;

class Frame{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static void Clear();

    Frame(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB,
        bool verbose = true);

    int static total_frame;

    int frame_seq_id;    // image topic sequence id, fixed
    cv::Mat frame_img;      // depth img for processing
    cv::Mat rgb_img;        // rgb img for visualization.
    cv::Mat gray_img;       // gray! for texture
    cv::Mat ellipsoids_2d_img;

    double timestamp;

    g2o::VertexSE3Expmap* pose_vertex;

    g2o::SE3Quat cam_pose_Tcw;	     // optimized pose  world to cam
    g2o::SE3Quat cam_pose_Twc;	     // optimized pose  cam to world

    Eigen::MatrixXd mmObservations;     // id x1 y1 x2 y2 label rate instanceID
    std::vector<bool> mvbOutlier;

    // For depth ellipsoid extraction.
    bool mbHaveLocalObject;
    std::vector<g2o::ellipsoid*> mpLocalObjects; // local 3d ellipsoid

    // Store observations  ZHJD：这里存储的是2d观测和3d观测的结构体
    Measurements meas;

    // Store relations
    bool mbSetRelation;
    Relations relations;
};

}
#endif //ELLIPSOIDSLAM_FRAME_H
