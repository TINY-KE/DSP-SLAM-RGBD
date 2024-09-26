/**
* This file is part of https://github.com/JingwenWang95/DSP-SLAM
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#include "Tracking.h"
#include "ObjectDetection.h"
#include "ORBmatcher.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace ORB_SLAM2 {

    // 新版本则基于bbox生成，不再与RGBD版本有任何关联
    // replace_detection: 是否将推测物体放入 pFrame 中生效。
    void Tracking::InferObjectsWithSemanticPrior(Frame* pFrame, bool use_input_pri = true, bool replace_detection = false)
    {
        // 要求有地平面估计再启动
        if(miGroundPlaneState != 2)
        {
            std::cout << "Close Infering, as the groundplane is not set." << std::endl;
            return;
        }

        // ********* 测试1： 所有先验都是 1:1:1 *********
        // // 读取 pri;  调试模式从全局 Config 中读取
        // double d = Config::ReadValue<double>("SemanticPrior.PriorD");
        // double e = Config::ReadValue<double>("SemanticPrior.PriorE");
        double weight = Config::ReadValue<double>("SemanticPrior.Weight");
        // Pri pri(d,e);

        // std::cout << "Begin infering ... " << std::endl;
        // pri.print();

        // 对于帧内每个物体，做推断，并可视化新的物体
        auto& meas = pFrame->meas;
        int meas_num = meas.size();

        if(replace_detection) {
            pFrame->mpLocalObjects.clear();
            pFrame->mpLocalObjects.resize(meas_num);
        }
        for(int i=0;i<meas_num;i++)
        {
            Measurement& m = meas[i];
            Vector4d bbox = m.ob_2d.bbox;

            // Check : 确保该物体类型是在地面之上的
            if(!CheckLabelOnGround(m.ob_2d.label)) continue;

            // Check : 该 bbox 不在边缘
            bool is_border = calibrateMeasurement(bbox, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            if(is_border) continue;

            // 生成Pri

            // 设置Pri是否是读取或是默认
            Pri pri;
            if(use_input_pri)
                pri = mPrifac.CreatePri(m.ob_2d.label);
            else
                pri = Pri(1,1);
            
            // bool result_pri = GeneratePriFromLabel(m.ob_2d.label, pri);
            // // 如果不成功，我们按 1:1:1 计算?
            // if(!result_pri)
            // {
            //     std::cout << "No pri found for : " << m.ob_2d.label << std::endl;
            //     pri = Pri(1,1);
            // }

            std::cout << "Pri for label : " << m.ob_2d.label << std::endl;
            pri.print();

            // RGB_D + Prior
            g2o::plane ground_pl_local = mGroundPlane; ground_pl_local.transform(pFrame->cam_pose_Tcw);

            priorInfer pi(mRows, mCols, mCalib);
            // *********************************
            // 生成一个新的 Initguess
            // *********************************
            g2o::ellipsoid e_init_guess = pi.GenerateInitGuess(bbox, ground_pl_local.param);
            // ------------------------

            bool bUsePriInit = false;   // 未开发完成的功能
            g2o::ellipsoid e_infer_mono_guess;
            if(bUsePriInit){
                e_infer_mono_guess = pi.MonocularInferExpand(e_init_guess, pri, weight, ground_pl_local);

                // DEBUG 可视化 挨个显示所有的椭球体
                // auto vEsForVisual = pi.GetAllPossibleEllipsoids();
                // for(int i=0;i<vEsForVisual.size();i++)
                // {
                //     g2o::ellipsoid* pETemp = new g2o::ellipsoid(vEsForVisual[i].transform_from(pFrame->cam_pose_Twc));
                //     // mpMap->ClearEllipsoidsVisual();
                //     pETemp->setColor(Vector3d(1.0,0,0));
                //     mpMap->addEllipsoidVisual(pETemp);
                //     std::cout << "Visualize " << i << std::endl;
                //     std::cout << "Wait for push ... " << std::endl;

                //     bool bDebug = false;
                //     if(bDebug)
                //         getchar();
                // }
            }
            else 
                e_infer_mono_guess = pi.MonocularInfer(e_init_guess, pri, weight, ground_pl_local);
            // 设置椭球体label, prob
            e_infer_mono_guess.miLabel = m.ob_2d.label;
            e_infer_mono_guess.prob = m.ob_2d.rate; // 暂时设置为 bbox 检测的概率吧
            e_infer_mono_guess.bbox = m.ob_2d.bbox;
            e_infer_mono_guess.prob_3d =  1.0; // 暂定!
            g2o::ellipsoid* pEInfer_mono_guess = new g2o::ellipsoid(e_infer_mono_guess.transform_from(pFrame->cam_pose_Twc));

            Vector3d color_rgb(144,238,144); color_rgb/=255.0;
            if(!use_input_pri) color_rgb = Vector3d(1,0,0); // 默认版本为红色
            pEInfer_mono_guess->setColor(color_rgb);
            mpMap->addEllipsoidObservation(pEInfer_mono_guess); // 可视化
            std::cout << " Before Monocular Infer: " << e_init_guess.toMinimalVector().transpose() << std::endl;
            std::cout << " After Monocular Infer: " << e_infer_mono_guess.toMinimalVector().transpose() << std::endl;

            // DEBUG可视化： 显示一下初始状态的椭球体
            // g2o::ellipsoid* pE_init_guess = new g2o::ellipsoid(e_init_guess.transform_from(pFrame->cam_pose_Twc));
            // pE_init_guess->prob = 1.0;
            // pE_init_guess->setColor(Vector3d(0.1,0,0.1));
            // mpMap->addEllipsoidVisual(pE_init_guess); // 可视化

            // --------- 将结果放到frame中存储
            if(replace_detection)
                pFrame->mpLocalObjects[i] = new g2o::ellipsoid(e_infer_mono_guess);

            // DEBUGING: 调试为何Z轴会发生变化， 先输出在局部坐标系下的两个rotMat
            // std::cout << "InitGuess RotMat in Camera: " << std::endl << e_init_guess.pose.rotation().toRotationMatrix() << std::endl;
            // std::cout << "Infered RotMat in Camera: " << std::endl << e_infer_mono_guess.pose.rotation().toRotationMatrix() << std::endl;
            // std::cout << "GroundPlaneNorma in Camera: " << std::endl << ground_pl_local.normal().head(3).normalized() << std::endl;

            // 可视化bbox的约束平面
            VisualizeConstrainPlanes(e_infer_mono_guess, pFrame->cam_pose_Twc, mpMap); // 中点定在全局坐标系

        }       

        std::cout << "Finish infering for " << meas_num << " objects..." << std::endl;
        return;
    }


}