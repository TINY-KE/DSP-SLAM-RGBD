// Update: 19-12-31 by lzw
// 测试录制的真实椅子数据集的效果.

// Update: 19-12-24 by lzw
// 从ros话题读取检测结果测试椭球体重建效果.

// 该文件作为基本ros模板.

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/ObjectCount.h>
using namespace sensor_msgs;

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

// 多帧数据同步
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
using namespace message_filters;

#include <Eigen/Core>

#include <src/symmetry/Symmetry.h>
#include <include/core/Geometry.h>

#include <include/core/System.h>

#include <src/config/Config.h>

#include <src/pca/EllipsoidExtractor.h>

using namespace std;
using namespace QuadricSLAM;

class ImageGrabber
{
public:
    ImageGrabber(camera_intrinsic &camera, System* pSLAM):mCamera(camera), mpSLAM(pSLAM), mbFirstFrame(true){
        mpExt = new EllipsoidExtractor;
        mpEllipsoidOneShot = new g2o::ellipsoid();
        pSLAM->getMap()->addEllipsoidVisual(mpEllipsoidOneShot);  // 直接放入不行，会让optimization 开始工作

        mGlobalEllipsoid = new g2o::ellipsoid;
        mGlobalEllipsoid->setColor(Vector3d(0,1,0));    // green
        // mpSLAM->getMap()->addEllipsoidVisual(mGlobalEllipsoid); // 防止遮挡 先隐去.

        std::cout << "Attention: Point Cloud is filtered in an ellipsoid. " << std::endl;

        // 添加滚动条.
        id_ex = mpSLAM->getViewer()->addDoubleMenu("e.x", -2, 2, 1.117);
        id_ey = mpSLAM->getViewer()->addDoubleMenu("e.y", -2, 2, -0.135);
        id_half_size = mpSLAM->getViewer()->addDoubleMenu("e.half_size", 0.3, 2, 0.4252);

        id_scale_x = mpSLAM->getViewer()->addDoubleMenu("e.id_scale_x", 0.3, 2, 0.4147);
        id_scale_y = mpSLAM->getViewer()->addDoubleMenu("e.id_scale_y", 0.3, 2, 0.3);
        id_yaw = mpSLAM->getViewer()->addDoubleMenu("e.id_yaw", -M_PI, M_PI, 0.4818);

    }
    
    void GrabImage(const ImageConstPtr& depth, 
            const darknet_ros_msgs::BoundingBoxesConstPtr& bbox);
    
    VectorXd odomToCamPose(VectorXd & pose);
    // void GrabCameraInfo(camera_in);

    void updateEllipsoid(){

        mpSLAM->getViewer()->getValueDoubleMenu(id_ex, e_x);
        mpSLAM->getViewer()->getValueDoubleMenu(id_ey, e_y);
        mpSLAM->getViewer()->getValueDoubleMenu(id_half_size, e_half_size);

        mpSLAM->getViewer()->getValueDoubleMenu(id_scale_x, e_scale_x);
        mpSLAM->getViewer()->getValueDoubleMenu(id_scale_y, e_scale_y);
        mpSLAM->getViewer()->getValueDoubleMenu(id_yaw, e_yaw);

        Vector9d ellipsoid_vec; 
        ellipsoid_vec << e_x,e_y,e_half_size,0,0,e_yaw,e_scale_x,e_scale_y,e_half_size;     // 3自由度任意调节?
        mGlobalEllipsoid->fromMinimalVector(ellipsoid_vec);
    }

private:
    bool JudgeKeyFrame(VectorXd& pose);

//    ORB_SLAM2::System* mpSLAM;
private:
    camera_intrinsic mCamera;
    System* mpSLAM;

    VectorXd mLastPose;
    bool mbFirstFrame;

    EllipsoidExtractor* mpExt;
    g2o::ellipsoid* mpEllipsoidOneShot;

    g2o::ellipsoid* mGlobalEllipsoid;   // 帮助提取点云. debug 用. 
    double e_x,e_y,e_half_size, e_scale_x, e_scale_y, e_yaw;
    int id_ex, id_ey, id_half_size, id_scale_x, id_scale_y, id_yaw;

};

VectorXd getCameraGlobalPose()
{
    // 产生初始相机位姿
    Vector6d pose_xyzrpy;
    pose_xyzrpy << 0,0,0,0,0,0;

    // x, y, z, roll, pitch, yaw
    g2o::SE3Quat campose_cw; campose_cw.fromXYZPRYVector(pose_xyzrpy);

    VectorXd output = campose_cw.toVector();
    return output;

}

VectorXd ImageGrabber::odomToCamPose(VectorXd& pose)
{
    g2o::SE3Quat Tw_r;  // 机器人在世界的位置
    Tw_r.fromVector(pose.head(7));
    Matrix4d matTwr = Tw_r.to_homogeneous_matrix();

    Matrix4d matTrc;
    Vector3d trans(0,0,0.500);
    matTrc.block(0,3,3,1) = trans;

    AngleAxisd V1(-M_PI/2, Vector3d(0, 0, 1));//以（0,0,1）为旋转轴，旋转45度
    AngleAxisd V2(-M_PI / 2, Vector3d(1, 0, 0));//以（0,0,1）为旋转轴，旋转45度
    
    Matrix3d matRot = V1.matrix() * V2.matrix();
    matTrc.block(0,0,3,3) = matRot;
    matTrc.block(3,0,1,4) = Vector4d(0,0,0,1).transpose();

    // // 在此输入相机外参数.
    // g2o::Vector6d xyzrpy;
    // xyzrpy[0] = 0.005;
    // xyzrpy[1] = 0.018;
    // xyzrpy[2] = 0.800;
    // xyzrpy[3] = CV_PI/2.0;
    // xyzrpy[4] = 0;
    // xyzrpy[5] = -CV_PI/2.0;
    // Tr_c.fromXYZPRYVector(xyzrpy);  // 不确定这个怎么算的.

    Matrix4d matTwc = matTwr * matTrc;
    Matrix3d rot = matTwc.block(0,0,3,3);
    Quaterniond q(rot);

    // mat 如何 转 pose. rot 如何转quat.
    VectorXd campose; campose.resize(7,1);
    campose.head(3) = matTwc.block(0,3,3,1);
    Vector4d q_vec; q_vec << q.x(),q.y(),q.z(),q.w();
    campose.tail(4) = q_vec;

    // std::cout << "matTrc: " << endl << matTrc << std::endl;
    // std::cout << "campose : " << campose.transpose() << endl;

    return campose;

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SymmetryDetector");
    ros::start();

    if(argc < 2)
    {
        std::cout << "usage: " << argv[0] << " path_to_settings" << std::endl;
        ros::shutdown();
        return 1;
    }    

    Config::Init();
    Config::SetValue("Tracking_MINIMUM_INITIALIZATION_FRAME", 5);       

    camera_intrinsic camera; // Real Kinect内参数. - Company's
    camera.fx = 1081.372070;
    camera.fy = 1081.372070;
    camera.cx = 959.500000;
    camera.cy = 539.500000;
    camera.scale = 1000;

    string strSettingPath = string(argv[1]);
    System* pSLAM = new System(strSettingPath, true);

    // 开启对称性检测
    // pSLAM->OpenSymmetry();
    pSLAM->OpenDepthEllipsoid();

    ImageGrabber igb(camera, pSLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<Image> depth_sub(nh, "/camera/depth/image", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bbox_sub(nh, "/darknet_ros/bounding_boxes", 1);

    typedef message_filters::sync_policies::ApproximateTime
            <Image, darknet_ros_msgs::BoundingBoxes> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), depth_sub, bbox_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabImage,&igb,_1,_2));

    // 显示窗口
    cv::namedWindow("depth", cv::WINDOW_NORMAL);

    while(1)
    {
        ros::spinOnce();
        cv::waitKey(30);
        igb.updateEllipsoid();
    }

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const ImageConstPtr& msgDepth, const darknet_ros_msgs::BoundingBoxesConstPtr& msgBbox)
{
    cv::Mat imDepth, imDepth16U;
    VectorXd cam_pose, pose;
    Vector4d bbox;
   
    cv_bridge::CvImageConstPtr cv_ptrDepth;
    try
    {
        cv_ptrDepth = cv_bridge::toCvCopy(msgDepth);
        imDepth16U = cv_ptrDepth->image;
        // imDepth32F.convertTo(imDepth, CV_16UC1, 1000);
        imDepth = imDepth16U;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    auto boxes = msgBbox->bounding_boxes;
    if( boxes.size() == 0 ) 
    {
        std::cerr << "No detection. " << std::endl;
        return;
    }

    // 寻找 chair ,  id = 56
    bool find_chair = false;
    for( auto box : boxes )
    {
        if(box.id == 56 ){
            bbox << box.xmin,box.ymin,box.xmax,box.ymax;
            find_chair = true;
            break;
        }
    }

    if( !find_chair ){
        std::cerr << " No chair in detection. " << std::endl;
        return;
    }

    
    pose = getCameraGlobalPose();
    cam_pose = odomToCamPose(pose);
    std::cout << "Get Image (rows,cols) : " << imDepth.rows << "," << imDepth.cols << std::endl;
    std::cout << "Bbox: " << bbox.transpose() << std::endl;
    std::cout << "pose: " << pose.transpose() << std::endl;
    std::cout << "cam_pose: " << cam_pose.transpose() << std::endl;

    // 可视化
    // cv::Mat imShow = imRGB.clone();
    // cv::Rect rec(cv::Point(bbox(0), bbox(1)), cv::Point(bbox(2), bbox(3)));
    // cv::rectangle(imShow, rec, cv::Scalar(255,0,0), 3);
    // // cv::imshow("rgb", imShow);    
    cv::imshow("depth", imDepth);    
    cv::waitKey(10);

    // ------------- 正式处理 -----------------

    // 关键帧判断
    bool isKeyFrame = true;

    if( isKeyFrame )
    {
        MatrixXd bboxMat;bboxMat.resize(1, 8);
        bboxMat.row(0)[0] = 0; // id
        bboxMat.block(0,1,1,4) = bbox.transpose();
        bboxMat.row(0)[5] = 1; //label
        bboxMat.row(0)[6] = 1; //Rate
        bboxMat.row(0)[7] = 1; //label

        std::cout << "bboxMat: " << std::endl << bboxMat << std::endl;

        // bboxMat: id x1 y1 x2 y2 label rate [Instance]
        // mpSLAM->TrackWithObjects(cam_pose, bboxMat, imDepth, true);   // 其中 imRGB只用作可视化.

        // 处理对称性.
        QuadricSLAM::SymmetryOutputData data;
        QuadricSLAM::Symmetry sym;
        sym.SetEllipsoid(*mGlobalEllipsoid);
        data = sym.detectSymmetryPlaneSparse(imDepth, bbox, cam_pose, mCamera, 2.0);
        // 点云可视化.
        if( data.result )
        {
            QuadricSLAM::Map* mpMap = mpSLAM->getMap();
            mpMap->clearPointCloud();
            mpMap->clearPlanes();

            SetPointCloudProperty(data.pCloud, 0,0,0,10);
            mpMap->addPointCloud(data.pCloud);

            g2o::plane* pPlane = new g2o::plane(data.planeVec, Vector3d(0,0,1.0));
            mpMap->addPlane(pPlane);

            g2o::SE3Quat* pCampose_wc = new g2o::SE3Quat; 
            pCampose_wc->fromVector(cam_pose);
            mpMap->setCameraState(pCampose_wc);
            mpMap->addCameraStateToTrajectory(pCampose_wc);

            // 基于提取的点云估计椭球体
            pcl::PointCloud<PointType>::Ptr pCloudPCL = QuadricPointCloudToPclXYZ(*data.pCloud);
            g2o::ellipsoid e_exted = mpExt->EstimateEllipsoidFromPointCloud(pCloudPCL);
            if( mpExt->GetResult() )
            {
                // 更新该椭球体.
                Vector3d color_vec(1,0,0);
                e_exted.setColor(color_vec);
                (*mpEllipsoidOneShot) = e_exted;
            }
            
        }
    }

}

