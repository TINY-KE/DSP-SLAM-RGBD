#ifndef ELLIPSOIDSLAM_ELLIPSOID_H_zhjd
#define ELLIPSOIDSLAM_ELLIPSOID_H_zhjd

// #include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
// #include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "include/utils/matrix_utils.h"
 
using namespace Eigen;

typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 3, 8> Matrix38d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;
typedef Eigen::Matrix<double, 6, 1> Vector7d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

namespace g2o
{

// class plane;
// class ConstrainPlane;
class ellipsoid_zhjd
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // SE3Quat pose;  // rigid body transformation, object in world coordinate
    Eigen::Vector3d scale; // a,b,c : half length of axis x,y,z

    Vector9d vec_minimal; // x,y,z,roll,pitch,yaw,a,b,c

    double prob;    // probability from single-frame ellipsoid estimation
    double prob_3d; // 椭球体三维投影到平面上，与bbox的IoU比较结果

    bool bPointModel;

    int miLabel;        // semantic label.
    int miInstanceID;   // instance id.

    Vector4d bbox;  // Local局部观测中存储它.

    Eigen::MatrixXd cplanes;   // constrain 3d planes Nx4, one plane each ROW.
    // std::vector<ConstrainPlane*> mvCPlanes;
    // std::vector<ConstrainPlane*> mvCPlanesWorld;

    ellipsoid_zhjd();
    // Copy constructor.
    // ellipsoid(const ellipsoid &e);
    bool mbColor;

};

} // g2o

#endif