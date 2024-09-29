#include "include/ellipsoid/Ellipsoid_zhjd.h"

#include "src/Polygon/Polygon.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>

// #include "include/core/Plane.h"
// #include "include/core/ConstrainPlane.h"
#include <boost/math/distributions/chi_squared.hpp>

// 卡方分布的分布函数
double chi2cdf(int degree, double chi)
{
    boost::math::chi_squared mydist(degree);
    double p = boost::math::cdf(mydist,chi);
    return p;
}

namespace g2o
{
    ellipsoid_zhjd::ellipsoid_zhjd():miInstanceID(-1),mbColor(false),bPointModel(false)
    {
    }


    // xyz quaternion, half_scale
    // void ellipsoid::fromVector(const Vector10d& v){
    //     std::cout << "e_param: 2:" << v.transpose() << std::endl;
        
    //     // pose.fromVector(v.head<7>());
    //     std::cout << "e_param: 3:" << std::endl;

    //     scale = v.tail<3>();
    //     std::cout << "e_param: 4:" << scale.transpose() << std::endl;
    //     // vec_minimal = toMinimalVector();
    //     std::cout << "e_param: 5:" << vec_minimal.transpose() << std::endl;
    // }


} // g2o