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

string DatasetType;

class ImageGrabber
{
public:
    ImageGrabber(camera_intrinsic &camera, System* pSLAM):mCamera(camera), mpSLAM(pSLAM), mbFirstFrame(true){
        mpExt = new EllipsoidExtractor;
        // mpEllipsoidOneShot = new g2o::ellipsoid();
        // pSLAM->getMap()->addEllipsoid(mpEllipsoidOneShot);  // 直接放入不行，会让optimization 开始工作
    }

    ImageGrabber(System* pSLAM):mpSLAM(pSLAM), mbFirstFrame(true){
        mpExt = new EllipsoidExtractor;
    }
    
    void GrabImage(const CompressedImageConstPtr& image, const ImageConstPtr& depth, 
            const nav_msgs::OdometryConstPtr& odom, const darknet_ros_msgs::BoundingBoxesConstPtr& bbox);
    
    VectorXd odomToCamPose(VectorXd & pose);
    // void GrabCameraInfo(camera_in);

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
    Config::SetValue("Tracking_MINIMUM_INITIALIZATION_FRAME", 5); 
    Config::SetValue("EllipsoidExtractor_PLANE_HEIGHT", 0);       // 设置全局平面高度
    Config::SetValue("EllipsoidExtractor_DEPTH_RANGE", 10);  

    string strSettingPath = string(argv[1]);
    System* pSLAM = new System(strSettingPath, true);

    // 开启对称性检测
    // pSLAM->OpenSymmetry();
    pSLAM->OpenDepthEllipsoid();
    pSLAM->getTracker()->OpenGroundPlaneEstimation();

    ImageGrabber igb(pSLAM);

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
    // cv::namedWindow("rgb", cv::WINDOW_NORMAL);
    // cv::namedWindow("depth", cv::WINDOW_NORMAL);

    DatasetType = Config::Get<string>("Dataset.Type");
    std::cout << "Dataset Type: " << DatasetType << std::endl;

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

    if( DatasetType == "Gazebo")
        cam_pose = odomToCamPose(pose); // 使用外参数获得相机位姿
    else
        cam_pose = pose;    // 保持原样.

    auto boxes = msgBbox->bounding_boxes;
    if( boxes.size() == 0 ) 
    {
        std::cerr << "No detection. " << std::endl;
        return;
    }
    
    bbox << boxes[0].xmin,boxes[0].ymin,boxes[0].xmax,boxes[0].ymax;

    // std::cout << "Get Image (rows,cols) : " << imDepth.rows << "," << imDepth.cols << std::endl;
    // std::cout << "Bbox: " << bbox.transpose() << std::endl;
    // std::cout << "pose: " << pose.transpose() << std::endl;
    // std::cout << "cam_pose: " << cam_pose.transpose() << std::endl;

    // 可视化
    // cv::Mat imShow = imRGB.clone();
    // cv::Rect rec(cv::Point(bbox(0), bbox(1)), cv::Point(bbox(2), bbox(3)));
    // cv::rectangle(imShow, rec, cv::Scalar(255,0,0), 3);
    // // cv::imshow("rgb", imShow);    
    // cv::imshow("depth", imDepth);    
    // cv::waitKey(10);

    // ------------- 正式处理 -----------------

    // 关键帧判断
    bool isKeyFrame = JudgeKeyFrame(cam_pose);

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
        mpSLAM->TrackWithObjects(cam_pose, bboxMat, imDepth, imRGB, true);   // 其中 imRGB只用作可视化.

        // // 测试新工具.
        // g2o::ellipsoid e_exted = mpExt->EstimateEllipsoid(imDepth, bbox, cam_pose, mCamera);
        // if( mpExt->GetResult() )
        // {
        //     // 更新该椭球体.
        //     Vector3d color_vec(0,1,0);
        //     e_exted.setColor(color_vec);
        //     (*mpEllipsoidOneShot) = e_exted;
        // }

        // // 处理对称性.
        // QuadricSLAM::SymmetryOutputData data;
        // QuadricSLAM::Symmetry sym;
        // data = sym.detectSymmetryPlaneSparse(imDepth, bbox, cam_pose, mCamera);
        // // 点云可视化.
        // if( data.result )
        // {
        //     QuadricSLAM::Map* mpMap = mpSLAM->getMap();
        //     mpMap->clearPointCloud();
        //     mpMap->clearPlanes();

        //     mpMap->addPointCloud(data.pCloud);

        //     // g2o::plane* pPlane = new g2o::plane(data.planeVec, Vector3d(0,0,1.0));
        //     // mpMap->addPlane(pPlane);

        //     // g2o::SE3Quat* pCampose_wc = new g2o::SE3Quat; 
        //     // pCampose_wc->fromVector(cam_pose);
        //     // mpMap->setCameraState(pCampose_wc);
        //     // mpMap->addCameraStateToTrajectory(pCampose_wc);
            
        // }
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

    // std::cout << "matTrc: " << endl << matTrc << std::endl;
    // std::cout << "campose : " << campose.transpose() << endl;

    return campose;

}

bool ImageGrabber::JudgeKeyFrame(VectorXd &pose)
{
    if(mbFirstFrame )
    {
        mbFirstFrame = false;
        mLastPose = pose;
        return true;
    }

    double PARAM_MINIMUM_DIS = 0.1; // 最小移动距离差
    Eigen::Vector3d xyz = pose.head(3);
    Eigen::Vector3d xyz_last = mLastPose.head(3);
    double dis = std::abs(xyz(0) - xyz_last(0));

    // std::cout << "dis : " << dis << std::endl;

    if( dis > PARAM_MINIMUM_DIS )
    {
        mLastPose = pose;    
        return true;
    }

    else
        return false;

}