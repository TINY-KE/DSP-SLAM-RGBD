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

        Pri pri = Pri(1,1);
        pri.print();

        // ********* 测试1： 所有先验都是 1:1:1 *********
        // // 读取 pri;  调试模式从全局 Config 中读取
        double weight = Config::ReadValue<double>("SemanticPrior.Weight");
        std::cout << "weight:"<<weight << std::endl;
        std::cout << "Begin infering ... " << std::endl;

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
            // if(!CheckLabelOnGround(m.ob_2d.label)) continue;

            // Check : 该 bbox 不在边缘
            // bool is_border = calibrateMeasurement(bbox, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            // if(is_border) continue;

            // 生成Pri
            Pri pri = Pri(1,1);

            std::cout << "Pri for label : " << m.ob_2d.label << std::endl;
            pri.print();

            // RGB_D + Prior
            g2o::plane ground_pl_local = mGroundPlane; ground_pl_local.transform(pFrame->cam_pose_Tcw);

            priorInfer pi(mRows, mCols, mCalib);

            // *********************************
            // 生成一个新的 Initguess
            // *********************************
            std::cout<<"[GenerateInitGuess] 1 准备 初始化一个椭球体"<<std::endl;
            std::cout << "LastCost: " << pi.GetLastCost() << std::endl;
            // pi.GenerateInitGuess(bbox, ground_pl_local.param);
            // g2o::ellipsoid e_init_guess = pi.GenerateInitGuess(bbox, ground_pl_local.param);
            g2o::ellipsoid e_init_guess; // 获得估计出椭球体的 rpy; 只是将 x,y,z,a,b,c 都设置为0.
            Eigen::Matrix<double, 10, 1>  e_param;
            e_param <<     0, 0, 0,   // x y z
                            0, 0, 0, 0,  // qx qy qz qw
                            0.5, 0.5, 0.5   // length_a  length_b  length_c
                        ;
            std::cout << "e_param: 11:" << e_param.transpose() << std::endl;
            // e_init_guess.fromVector(e_param);
            std::cout<<"[GenerateInitGuess] 3 结束 初始化一个椭球体"<<std::endl;

            // g2o::ellipsoid e_init_guess = pi.GenerateInitGuess(bbox, ground_pl_local.param);
            // ------------------------

        //     bool bUsePriInit = false;   // 未开发完成的功能
        //     g2o::ellipsoid e_infer_mono_guess;
            
        //     e_infer_mono_guess = pi.MonocularInfer(e_init_guess, pri, weight, ground_pl_local);
        //     // 设置椭球体label, prob
        //     e_infer_mono_guess.miLabel = m.ob_2d.label;
        //     e_infer_mono_guess.prob = m.ob_2d.rate; // 暂时设置为 bbox 检测的概率吧
        //     e_infer_mono_guess.bbox = m.ob_2d.bbox;
        //     e_infer_mono_guess.prob_3d =  1.0; // 暂定!
        //     g2o::ellipsoid* pEInfer_mono_guess = new g2o::ellipsoid(e_infer_mono_guess.transform_from(pFrame->cam_pose_Twc));

        //     Vector3d color_rgb(144,238,144); color_rgb/=255.0;
        //     if(!use_input_pri) color_rgb = Vector3d(1,0,0); // 默认版本为红色
        //     pEInfer_mono_guess->setColor(color_rgb);
        //     // mpMap->addEllipsoidObservation(pEInfer_mono_guess); // 可视化
        //     std::cout << " Before Monocular Infer: " << e_init_guess.toMinimalVector().transpose() << std::endl;
        //     std::cout << " After Monocular Infer: " << e_infer_mono_guess.toMinimalVector().transpose() << std::endl;

        //     // DEBUG可视化： 显示一下初始状态的椭球体
        //     // g2o::ellipsoid* pE_init_guess = new g2o::ellipsoid(e_init_guess.transform_from(pFrame->cam_pose_Twc));
        //     // pE_init_guess->prob = 1.0;
        //     // pE_init_guess->setColor(Vector3d(0.1,0,0.1));
        //     // mpMap->addEllipsoidVisual(pE_init_guess); // 可视化

        //     // --------- 将结果放到frame中存储
        //     if(replace_detection)
        //         pFrame->mpLocalObjects[i] = new g2o::ellipsoid(e_infer_mono_guess);

        //     // DEBUGING: 调试为何Z轴会发生变化， 先输出在局部坐标系下的两个rotMat
        //     // std::cout << "InitGuess RotMat in Camera: " << std::endl << e_init_guess.pose.rotation().toRotationMatrix() << std::endl;
        //     // std::cout << "Infered RotMat in Camera: " << std::endl << e_infer_mono_guess.pose.rotation().toRotationMatrix() << std::endl;
        //     // std::cout << "GroundPlaneNorma in Camera: " << std::endl << ground_pl_local.normal().head(3).normalized() << std::endl;

        //     // 可视化bbox的约束平面
        //     // VisualizeConstrainPlanes(e_infer_mono_guess, pFrame->cam_pose_Twc, mpMap); // 中点定在全局坐标系

        }       

        std::cout << "Finish infering for " << meas_num << " objects..." << std::endl;
        return;
    }




    // [改进]
    // TODO: 生成椭球体
    void Tracking::UpdateObjectObservation_GenerateEllipsoid(ORB_SLAM2::Frame *pFrame, KeyFrame* pKF, bool withAssociation) {
        clock_t time_0_start = clock();

        // 
        bool use_infer_detection = Config::Get<int>("System.MonocularInfer.Open") > 0;
        use_infer_detection = true;

        // std::cout << "use_infer_detection = " << use_infer_detection << std::endl;

        // 思考: 如何让两个平面提取共用一个 MHPlane 提取.

        // // [0] 刷新可视化
        // ClearVisualization();

        // // [1] process MHPlanes estimation
        // TaskGroundPlane();
        
        // clock_t time_1_TaskGroundPlane = clock();

        // // New task : for Manhattan Planes
        // // TaskManhattanPlanes(pFrame);
        // clock_t time_2_TaskManhattanPlanes = clock();

        // [2] process single-frame ellipsoid estimation
        // clock_t time_3_UpdateDepthEllipsoidEstimation, time_4_TaskRelationship, time_5_RefineObjectsWithRelations;
        
        if(!use_infer_detection)
        {
            // //   使用深度图像估计物体椭球体
            // UpdateDepthEllipsoidEstimation(pFrame, pKF, withAssociation);
            // // time_3_UpdateDepthEllipsoidEstimation = clock();

            // // [3] Extract Relationship
            // //构建椭球体与曼哈顿平面之间的关联关系
            // TaskRelationship(pFrame);
            // // time_4_TaskRelationship = clock();

            // // // [4] Use Relationship To Refine Ellipsoids
            // // // 注意: Refine时必然在第一步可以初始化出有效的物体.
            // RefineObjectsWithRelations(pFrame);
            // // time_5_RefineObjectsWithRelations = clock();

            // // // [5] 对于第一次提取，Refine提取都失败的，使用点模型
            // // UpdateDepthEllipsoidUsingPointModel(pFrame);
            // GenerateObservationStructure(pFrame);
        }
        // [6] 补充调试环节： 测试语义先验对物体的影响
        else
        {

            // 将std::vector<ObjectDetection*>结构的mvpDetectedObjects 转为 Eigen::MatrixXd结构的mmObservations
            pFrame->GetObservations_fromKeyFrame(pKF);  

            GenerateObservationStructure(pFrame);   // 注意必须生成 measure 结构才能 Infer

            // const string priconfig_path = Config::Get<std::string>("Dataset.Path.PriTable");
            // bool bUseInputPri = (priconfig_path.size() > 0);
            // InferObjectsWithSemanticPrior(pFrame, bUseInputPri, use_infer_detection); // 使用PriTable先验初始化，然而存在问题，尚未调试完毕

            InferObjectsWithSemanticPrior(pFrame, false, use_infer_detection);   // 使用 1:1:1 的比例初始化

            GenerateObservationStructure(pFrame);
        }

        // // Output running time
        // // cout << " -- UpdateObjectObservation_GenerateEllipsoid Time: " << endl;
        // // cout << " --- time_1_TaskGroundPlane: " <<(double)(time_1_TaskGroundPlane - time_0_start) / CLOCKS_PER_SEC << "s" << endl;        
        // // cout << " --- time_2_TaskManhattanPlanes: " <<(double)(time_2_TaskManhattanPlanes - time_1_TaskGroundPlane) / CLOCKS_PER_SEC << "s" << endl;        
        // // cout << " --- time_3_UpdateDepthEllipsoidEstimation: " <<(double)(time_3_UpdateDepthEllipsoidEstimation - time_2_TaskManhattanPlanes) / CLOCKS_PER_SEC << "s" << endl;        
        // // cout << " --- time_4_TaskRelationship: " <<(double)(time_4_TaskRelationship - time_3_UpdateDepthEllipsoidEstimation) / CLOCKS_PER_SEC << "s" << endl;        
        // // cout << " --- time_5_RefineObjectsWithRelations: " <<(double)(time_5_RefineObjectsWithRelations - time_4_TaskRelationship) / CLOCKS_PER_SEC << "s" << endl;        

    }

    void Tracking::GenerateObservationStructure(ORB_SLAM2::Frame* pFrame)
    {
        // 本存储结构以物体本身观测为索引.
        // pFrame->meas;
        
        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int ob_num = obs_mat.rows();

        // 3d ob
        std::vector<g2o::ellipsoid*> pLocalObjects = pFrame->mpLocalObjects;

        // if(ob_num != pLocalObjects.size()) 
        // {
        //     std::cout << " [Error] 2d observations and 3d observations should have the same size." << std::endl;
        //     return;
        // }

        for( int i = 0; i < ob_num; i++)
        {
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate imageID
            int label = round(det_vec(5));
            Eigen::Vector4d bbox = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            Observation ob_2d;
            ob_2d.label = label;
            ob_2d.bbox = bbox;
            ob_2d.rate = det_vec(6);
            ob_2d.pFrame = pFrame;

            Observation3D ob_3d;
            ob_3d.pFrame = pFrame;
            if(pLocalObjects.size() == ob_num)
                ob_3d.pObj = pLocalObjects[i];
            else 
                ob_3d.pObj = NULL;

            Measurement m;
            m.measure_id = i;
            m.instance_id = -1; // not associated
            m.ob_2d = ob_2d;
            m.ob_3d = ob_3d;
            pFrame->meas.push_back(m);
        }
    }

    void Tracking::SetGroundPlaneMannually(const Eigen::Vector4d &param)
    {
        std::cout << "[GroundPlane] Set groundplane mannually: " << param.transpose() << std::endl;
        miGroundPlaneState = 3;
        mGroundPlane.param = param;
        mGroundPlane.color = Vector3d(0,1,0);
    }


    void Tracking::SetRealPose(ORB_SLAM2::Frame* pFrame){
        // std::cout << "[Set real pose for the first frame from] : "<< mStrSettingPath << std::endl;
        cv::FileStorage fSettings(mStrSettingPath, cv::FileStorage::READ);
        int ConstraintType = fSettings["ConstraintType"];
        if ( ConstraintType != 1 && ConstraintType != 2 && ConstraintType != 3){
            std::cerr << ">>>>>> [WARRNING] USE NO PARAM CONSTRAINT TYPE!" << std::endl;
            // ConstraintType = 1;
            std::exit(EXIT_FAILURE);  // 或者：std::abort();
        }
        if (ConstraintType == 1){// robot_camera tf
            float qx = fSettings["Tworld_camera.qx"], qy = fSettings["Tworld_camera.qy"], qz = fSettings["Tworld_camera.qz"], qw = fSettings["Tworld_camera.qw"],
                    tx = fSettings["Tworld_camera.tx"], ty = fSettings["Tworld_camera.ty"], tz = fSettings["Tworld_camera.tz"];
            //float qx = fSettings["Tgroud_firstcamera.qx"], qy = fSettings["Tgroud_firstcamera.qy"], qz = fSettings["Tgroud_firstcamera.qz"], qw = fSettings["Tgroud_firstcamera.qw"],
            //       tx = fSettings["Tgroud_firstcamera.tx"], ty = fSettings["Tgroud_firstcamera.ty"], tz = fSettings["Tgroud_firstcamera.tz"];
            mCurrentFrame.mGroundtruthPose_mat = cv::Mat::eye(4, 4, CV_32F);
            Eigen::Quaterniond quaternion(Eigen::Vector4d(qx, qy, qz, qw));
            Eigen::AngleAxisd rotation_vector(quaternion);
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            T.rotate(rotation_vector);
            T.pretranslate(Eigen::Vector3d(tx, ty, tz));
            Eigen::Matrix4d GroundtruthPose_eigen = T.matrix();
            cv::Mat cv_mat_32f;
            cv::eigen2cv(GroundtruthPose_eigen, cv_mat_32f);
            cv_mat_32f.convertTo(mCurrentFrame.mGroundtruthPose_mat, CV_32F);

        } else if(ConstraintType == 2){
            // TODO: IMU
        } else if (ConstraintType == 3){// ros tf
            // tf::TransformListener listener;
            // tf::StampedTransform transform;
            // cv::Mat T_w_camera = cv::Mat::eye(4,4,CV_32F);
            // try
            // {
            //     listener.waitForTransform("/map", "/camera_depth_optical_frame", ros::Time(0), ros::Duration(1.0));
            //     listener.lookupTransform("/map", "/camera_depth_optical_frame", ros::Time(0), transform);
            //     T_w_camera = Converter::Quation2CvMat(
            //                     transform.getRotation().x(),
            //                     transform.getRotation().y(),
            //                     transform.getRotation().z(),
            //                     transform.getRotation().w(),
            //                     transform.getOrigin().x(),
            //                     transform.getOrigin().y(),
            //                     transform.getOrigin().z()
            //             );
            // }
            // catch (tf::TransformException &ex)
            // {
            //     ROS_ERROR("%s -->> lost tf from /map to /base_footprint",ex.what());
            // }

            // mCurrentFrame.mGroundtruthPose_mat = T_w_camera;
        }

        // std::cout << "[Set real pose for the first frame from] : End" << std::endl;
    }



    void Tracking::ActivateGroundPlane(g2o::plane &groundplane)
    {
        // Set groundplane to EllipsoidExtractor
        // if( mbDepthEllipsoidOpened ){
        //     std::cout << " * Add supporting plane to Ellipsoid Extractor." << std::endl;
        //     mpEllipsoidExtractor->SetSupportingPlane(&groundplane, false);
        // }

        // Set groundplane to Optimizer
        std::cout << " * Add supporting plane to optimizer. " << std::endl;
        mpOptimizer->SetGroundPlane(groundplane.param);

        // std::cout << " * Add supporting plane to Manhattan Plane Extractor. " << std::endl;
        // pPlaneExtractorManhattan->SetGroundPlane(&groundplane);

        // Change state
        miGroundPlaneState = 2;
    }


    void Tracking::VisualizeManhattanPlanes()
    {
        // // 可视化提取结果.
        // std::vector<g2o::plane*> vDominantMHPlanes = pPlaneExtractorManhattan->GetDominantMHPlanes();
        // for(auto& vP:vDominantMHPlanes ) {
        //     mpMap->addPlane(vP);
        // }
    }

    

}