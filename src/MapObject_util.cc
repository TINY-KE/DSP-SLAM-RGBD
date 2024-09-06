//
// Created by zhjd on 24-4-12.
//
#include <MapObject.h>
#include "Converter.h"
#include <Eigen/Dense>

namespace ORB_SLAM2{
    void MapObject::compute_sdf_loss_of_all_inside_points(){
        // py::module optim  = py::module::import("reconstruct.optimizer");
        // pyOptimizer = optim.attr("Optimizer")(mpLocalMapper->mp, pSys->pyCfg);
        // py::module SDF_Loss  = py::module::import("reconstruct.loss");
        // pyLoss = SDF_Loss.attr("compute_sdf_loss_of_all_inside_points")(surface_points_cam, rays, depth_obs, pMO->vShapeCode);

        Eigen::MatrixXf surface_points_obj = Eigen::MatrixXf::Zero(GetMapPointsOnObject_foronlymono().size(), 3);
        int p_i = 0;
        for (auto pMP : GetMapPointsOnObject_foronlymono())
        {
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (pMP->isOutlier())
                continue;

            // 获得Row 和 tow
            auto Two = GetPoseSim3();
            cv::Mat Row = Converter::toCvMat(Two).rowRange(0,3).colRange(0,3);
            cv::Mat tow = Converter::toCvMat(Two).rowRange(0,3).col(3);;

            // pOP = T_object_map*pMP
            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Do = Row * x3Dw + tow;
            float xo = x3Do.at<float>(0);
            float yo = x3Do.at<float>(1);
            float zo = x3Do.at<float>(2);
            surface_points_obj(p_i, 0) = xo;
            surface_points_obj(p_i, 1) = yo;
            surface_points_obj(p_i, 2) = zo;
            p_i++;
        }

        auto pySdfLoss = pyOptimizer.attr("compute_sdf_loss_objectpoint_zhjd")
                (surface_points_obj /* 类型为Eigen::MatrixXf  */, this->vShapeCode);

        mdSdfLoss = py::cast<double>(pySdfLoss);

        std::cout<<"[mapobject sdf loss]cpp: "<<mdSdfLoss<<std::endl;
    }

    double MapObject::compute_sdf_loss(double x, double y, double z){

        Eigen::MatrixXf surface_points_obj = Eigen::MatrixXf::Zero(1, 3);

        //surface_points_obj(0, 0) = x;
        //surface_points_obj(0, 1) = y;
        //surface_points_obj(0, 2) = z;
        surface_points_obj(0, 0) = 0;
        surface_points_obj(0, 1) = 0;
        surface_points_obj(0, 2) = 0;
        std::cout<<"surface_points_obj"<<surface_points_obj<<std::endl;
        auto pySdfLoss = pyOptimizer.attr("compute_sdf_loss_objectpoint_zhjd")
                (surface_points_obj /* 类型为Eigen::MatrixXf  */, this->vShapeCode);

        double SdfLoss = py::cast<double>(pySdfLoss);
        std::cout<<"[rrt point sdf loss]cpp: "<<SdfLoss<<std::endl;

        return SdfLoss;
    }

    void MapObject::compute_NBV(){
        auto object_pose = GetPoseSim3();
        auto ox = object_pose(0,3);
        auto oy = object_pose(1,3);
        auto oz = object_pose(2,3);

        float px,py,pz;
        for (auto pMP : GetMapPointsOnObject_foronlymono()){
            if (!pMP)
                continue;
            if (pMP->isBad())
                continue;
            if (pMP->isOutlier())
                continue;
            // pOP = T_object_map*pMP
            cv::Mat x3Dw = pMP->GetWorldPos();
            px += x3Dw.at<float>(0);
            py += x3Dw.at<float>(1);
            pz += x3Dw.at<float>(2);
        }
        px/=GetMapPointsOnObject_foronlymono().size();
        py/=GetMapPointsOnObject_foronlymono().size();
        pz/=GetMapPointsOnObject_foronlymono().size();

        double scale = 3.0;
        double_t  nbvx = ox + scale*(ox - px);
        double_t  nbvy = oy + scale*(oy - py);
        double_t  nbvz = oz + scale*(oz - pz);

        Eigen::Vector3d start(nbvx,nbvy,nbvz);
        Eigen::Vector3d end(ox,oy,oz);

        nbv = new NBV(start,end);

//        cv::Mat pose_mat = Converter::toCvMat(nbv.pose_isometry3d);
    }
}