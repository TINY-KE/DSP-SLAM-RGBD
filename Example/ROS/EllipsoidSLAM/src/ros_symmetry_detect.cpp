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

using namespace std;
using namespace QuadricSLAM;

class ImageGrabber
{
public:
    ImageGrabber(camera_intrinsic &camera, System* pSLAM):mCamera(camera), mpSLAM(pSLAM){}
    
    void GrabImage(const CompressedImageConstPtr& image, const ImageConstPtr& depth, 
            const nav_msgs::OdometryConstPtr& odom, const darknet_ros_msgs::BoundingBoxesConstPtr& bbox);
    
    VectorXd odomToCamPose(VectorXd & pose);
    // void GrabCameraInfo(camera_in);

//    ORB_SLAM2::System* mpSLAM;
private:
    camera_intrinsic mCamera;
    System* mpSLAM;

};

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

    camera_intrinsic camera; // Gazebo Kinect内参数.
    camera.fx = 1206.8897719532354;
    camera.fy = 1206.8897719532354;
    camera.cx = 960.5;
    camera.cy = 540.5;
    camera.scale = 1000;

    string strSettingPath = string(argv[1]);
    System* pSLAM = new System(strSettingPath, true);

    ImageGrabber igb(camera, pSLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<CompressedImage> image_sub(nh, "/camera/rgb/image", 1);
    message_filters::Subscriber<Image> depth_sub(nh, "/camera/depth/image", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/odom", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bbox_sub(nh, "/darknet_ros/bounding_boxes", 1);

    typedef message_filters::sync_policies::ApproximateTime
            <CompressedImage, Image, nav_msgs::Odometry, darknet_ros_msgs::BoundingBoxes> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), image_sub, depth_sub, odom_sub, bbox_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabImage,&igb,_1,_2,_3,_4));

    // 显示窗口
    cv::namedWindow("rgb", cv::WINDOW_NORMAL);
    cv::namedWindow("depth", cv::WINDOW_NORMAL);

    ros::spin();

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const CompressedImageConstPtr& msgImage, const ImageConstPtr& msgDepth, 
            const nav_msgs::OdometryConstPtr& msgOdom, const darknet_ros_msgs::BoundingBoxesConstPtr& msgBbox)
{
    cv::Mat imRGB, imDepth, imDepth32F;
    VectorXd pose, cam_pose;
    Vector4d bbox;

  
    
    imRGB = cv::imdecode(cv::Mat(msgImage->data),1);//convert compressed image data to cv::Mat
    
    cv_bridge::CvImageConstPtr cv_ptrDepth;
    try
    {
        cv_ptrDepth = cv_bridge::toCvCopy(msgDepth);
        imDepth32F = cv_ptrDepth->image;
        imDepth32F.convertTo(imDepth, CV_16UC1, 1000);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // 获取里程计读数
    pose.resize(7, 1);
    pose << msgOdom->pose.pose.position.x,msgOdom->pose.pose.position.y,msgOdom->pose.pose.position.z,
            msgOdom->pose.pose.orientation.x,msgOdom->pose.pose.orientation.y,msgOdom->pose.pose.orientation.z,msgOdom->pose.pose.orientation.w;

    cam_pose = odomToCamPose(pose); // 使用外参数获得相机位姿

    auto boxes = msgBbox->bounding_boxes;
    if( boxes.size() == 0 ) 
    {
        std::cerr << "No detection. " << std::endl;
        return;
    }
    
    bbox << boxes[0].xmin,boxes[0].ymin,boxes[0].xmax,boxes[0].ymax;

    std::cout << "Get Image" << std::endl;
    std::cout << "Bbox: " << bbox.transpose() << std::endl;
    std::cout << "pose: " << pose.transpose() << std::endl;
    std::cout << "cam_pose: " << cam_pose.transpose() << std::endl;

    // 可视化
    cv::Mat imShow = imRGB.clone();
    cv::Rect rec(cv::Point(bbox(0), bbox(1)), cv::Point(bbox(2), bbox(3)));
    cv::rectangle(imShow, rec, cv::Scalar(255,0,0), 3);
    cv::imshow("rgb", imShow);    
    cv::imshow("depth", imDepth);    
    cv::waitKey(10);

    // ------------- 正式处理 -----------------

    QuadricSLAM::SymmetryOutputData data;
    // data = QuadricSLAM::Symmetry::detectSymmetryPlane(imDepth, bbox, cam_pose, mCamera);
    QuadricSLAM::Symmetry sym;
    data = sym.detectSymmetryPlaneSparse(imDepth, bbox, cam_pose, mCamera);
          
    // 可视化平面 - > 如何投影到2d图像?
    std::cout << "Prob : " << data.prob << std::endl;
    // std::cout << "Plane Vec: " << data.planeVec.transpose() << std::endl;

    // 点云可视化.
    if( data.result )
    {
        QuadricSLAM::Map* mpMap = mpSLAM->getMap();
        mpMap->clearPointCloud();
        mpMap->clearPlanes();

        mpMap->addPointCloud(data.pCloud);

        g2o::plane* pPlane = new g2o::plane(data.planeVec, Vector3d(0,0,1.0));
        mpMap->addPlane(pPlane);

        g2o::SE3Quat* pCampose_wc = new g2o::SE3Quat; 
        pCampose_wc->fromVector(cam_pose);
        mpMap->setCameraState(pCampose_wc);
        mpMap->addCameraStateToTrajectory(pCampose_wc);
        
    }
}

VectorXd ImageGrabber::odomToCamPose(VectorXd& pose)
{
    g2o::SE3Quat Tw_r;  // 机器人在世界的位置
    Tw_r.fromVector(pose.head(7));
    Matrix4d matTwr = Tw_r.to_homogeneous_matrix();

    Matrix4d matTrc;
    Vector3d trans(0.005,0.018,0.800);
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

    std::cout << "matTrc: " << endl << matTrc << std::endl;
    std::cout << "campose : " << campose.transpose() << endl;

    return campose;

}