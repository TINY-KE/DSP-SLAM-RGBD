//
// Created by zhjd on 24-4-13.
//

#ifndef DSP_SLAM_CANDIDATE_H
#define DSP_SLAM_CANDIDATE_H
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
#include "Converter.h"
#include <string>

namespace ORB_SLAM2{

class candidate{
    public:
        double centor_x,centor_y,centor_z;
        double roll,pitch,yaw;
        Eigen::Vector3d start;
        Eigen::Vector3d end;
        Eigen::Quaterniond oritention_quaterniond;
        Eigen::Isometry3d  pose_isometry3d;

    public:
        // 构造函数
        candidate(){
            centor_x = 0;
            centor_y = 0;
            centor_z = 0;
            roll = 0;
            pitch = 0;
            yaw = 0;
        }
        //This code creates a candidate trajectory for the robot to follow.
        candidate(Eigen::Vector3d start_, Eigen::Vector3d end_){
            start = start_;
            end = end_;

            Eigen::Vector3d direction = end - start;
            direction.normalize();

            Eigen::Quaterniond quaternion;
            std::string main_axix_type = "z";
            if(main_axix_type=="x")
                quaternion.setFromTwoVectors(Eigen::Vector3d::UnitX(), direction);
            else if(main_axix_type=="z"){
                quaternion.setFromTwoVectors(Eigen::Vector3d::UnitZ(), direction);
            }
            else if(main_axix_type=="x2z"){
                quaternion.setFromTwoVectors(Eigen::Vector3d::UnitX(), direction);
                Eigen::Matrix3d R_trans = (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY())
                                                    * Eigen::AngleAxisd(-1*M_PI_2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
                Eigen::Matrix3d R = quaternion.toRotationMatrix() * R_trans;
                quaternion = Eigen::Quaterniond(R);
            }
            else if(main_axix_type=="RPY_x2z"){
                 cv::Point3f v =  cv::Point3f(end.x(),  end.y(),  end.z()) - cv::Point3f(start.x(),  start.y(),  start.z());
                 yaw = std::atan2(v.y, v.x);  // 绕Z轴的旋转角度
                 pitch = std::atan2(-v.z, v.x);  // 绕y轴的旋转角度
                 // roll = std::atan2 因为要让相机视线（机械臂末端的x轴）与目标向量相同，有很多选择，因此我选择了限制roll=0
                 // 计算旋转矩阵
                 Eigen::Matrix3d rotation_matrix_;
                Eigen::Matrix<double,3,3> R;
                R << 1,0,0,0,1,0,0,0,1;
                Eigen::Vector3d rpy = R.eulerAngles(0, 1, 2);
//                std::cout << "Roll1: " << rpy[0] << std::endl;
//                std::cout << "Pitch: " << rpy[1] << std::endl;
//                std::cout << "Yaw: " << rpy[2] << std::endl;
                rotation_matrix_ = R * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
                rpy = rotation_matrix_.eulerAngles(0, 1, 2);
//                std::cout << "Roll2: " << rpy[0] << std::endl;
//                std::cout << "Pitch: " << rpy[1] << std::endl;
//                std::cout << "Yaw: " << rpy[2] << std::endl;
                 rotation_matrix_ = R * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
                                // * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
                rpy = rotation_matrix_.eulerAngles(0, 1, 2);
//                std::cout << "Roll3: " << rpy[0] << std::endl;
//                std::cout << "Pitch: " << rpy[1] << std::endl;
//                std::cout << "Yaw: " << rpy[2] << std::endl;
                Eigen::Matrix3d R_trans = (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY())
                             * Eigen::AngleAxisd(-1*M_PI_2, Eigen::Vector3d::UnitZ())).toRotationMatrix();
                 quaternion = Eigen::Quaterniond(rotation_matrix_*R_trans);
                // std::cout<<"Eigen::AngleAxisd: x "<<quaterniond_.x()
                //                             <<"  y "<<quaterniond_.y()
                //                             <<"  z "<<quaterniond_.z()
                //                             <<"  w "<<quaterniond_.w()<<std::endl;
            }

            oritention_quaterniond = quaternion;

            centor_x = start(0);
            centor_y = start(1);
            centor_z = start(2);
            pose_isometry3d = Eigen::Translation3d(centor_x, centor_y, centor_z) * oritention_quaterniond;

        }

        cv::Mat getCVMatPose(){
            return Converter::toCvMat(pose_isometry3d);
        }

    };

class NBV: public candidate{

    public:
        // 构造函数
        NBV(Eigen::Vector3d start_, Eigen::Vector3d end_)
                : candidate(start_, end_) {
        }

};

}


#endif //DSP_SLAM_CANDIDATE_H
