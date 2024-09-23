// Update: 2021-4-30 by Lzw
// 为 demo 而修改，融合 ORBSLAM2与YOLO的实时演示

// Update: 20-1-8 by lzw
// 适应真实数据集. 

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

#include <geometry_msgs/PoseStamped.h>
using namespace geometry_msgs;

#include <Eigen/Core>

#include <include/core/Geometry.h>
#include <include/core/System.h>
#include <include/core/Tracking.h>
#include <src/config/Config.h>

using namespace std;
using namespace EllipsoidSLAM;

std::string DatasetType;

class ImageGrabber
{
public:
    ImageGrabber(camera_intrinsic &camera, System* pSLAM):mCamera(camera), mpSLAM(pSLAM), mbFirstFrame(true){
        // mpExt = new EllipsoidExtractor;
        // mpEllipsoidOneShot = new g2o::ellipsoid();
        // pSLAM->getMap()->addEllipsoid(mpEllipsoidOneShot);  // 直接放入不行，会让optimization 开始工作
    }

    ImageGrabber(System* pSLAM):mpSLAM(pSLAM), mbFirstFrame(true){
        // mpExt = new EllipsoidExtractor;
    }
    
    void GrabImage(const ImageConstPtr& image, const ImageConstPtr& depth, 
            const geometry_msgs::PoseStampedConstPtr& msgOdom, const darknet_ros_msgs::BoundingBoxesConstPtr& bbox);
    
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
    if(argc < 2)
    {
        std::cout << "usage: " << argv[0] << " path_to_settings" << std::endl;
        return 1;
    }    

    // ros::start();

    // Config::Init();
    // Config::SetValue("Tracking_MINIMUM_INITIALIZATION_FRAME", 5); 
    // Config::SetValue("EllipsoidExtractor_PLANE_HEIGHT", 0);       // 设置全局平面高度
    // Config::SetValue("EllipsoidExtractor_DEPTH_RANGE", 6);  

    std::string strSettingPath = std::string(argv[1]);
    System* pSLAM = new System(strSettingPath, true);
    // System* pSLAM = NULL;

    // 开启对称性检测
    // pSLAM->OpenSymmetry();
    // pSLAM->OpenDepthEllipsoid();
    // pSLAM->getTracker()->OpenGroundPlaneEstimation();

    ImageGrabber igb(pSLAM);
    ros::init(argc, argv, "EllipsoidSLAM");
    ros::NodeHandle nh;

    message_filters::Subscriber<Image> image_sub(nh, "/camera/rgb/image", 1);
    message_filters::Subscriber<Image> depth_sub(nh, "/camera/depth/image", 1);
    message_filters::Subscriber<geometry_msgs::PoseStamped> odom_sub(nh, "/odom", 1);
    message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> bbox_sub(nh, "/darknet_ros/bounding_boxes", 1);

    typedef message_filters::sync_policies::ApproximateTime
            <Image, Image, geometry_msgs::PoseStamped, darknet_ros_msgs::BoundingBoxes> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), image_sub, depth_sub, odom_sub, bbox_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabImage,&igb,_1,_2,_3,_4));

    // 显示窗口
    // cv::namedWindow("rgb", cv::WINDOW_NORMAL);
    // cv::namedWindow("depth", cv::WINDOW_NORMAL);

    std::cout << "Waiting for comming frames..." << std::endl;

    ros::spin();
    ros::shutdown();

    return 0;
}

// bboxMat: id x1 y1 x2 y2 label rate [Instance]
MatrixXd rosMsgToMat(std::vector<darknet_ros_msgs::BoundingBox>& boxes)
{
    MatrixXd boxMat; boxMat.resize(boxes.size(), 7);
    for( int i=0;i<boxMat.rows();i++ )
    {
        auto box = boxes[i];
        Vector7d boxVec;
        boxVec << i,box.xmin,box.ymin,box.xmax,box.ymax,box.id,box.probability;
        boxMat.row(i) = boxVec.transpose();
    }
    return boxMat;
}

// 筛选留下指定label的物体检测
MatrixXd FilterDetectionMat(MatrixXd& matIn, std::set<int>& ignore_labels)
{
    MatrixXd matOut; matOut.resize(0, matIn.cols());
    int rows = matIn.rows();

    for( int i=0;i<rows;i++ )
    {
        VectorXd vec = matIn.row(i);
        int det_label = round(vec(5));

        // bool c1 =  select_labels.find(det_label) != select_labels.end();    // 选中label
        bool c1= true;

        bool c2 = ignore_labels.find(det_label) == ignore_labels.end();   // 不在忽略label内部.

        if( c1 && c2 )
        {
            matOut.conservativeResize(matOut.rows()+1, matOut.cols());
            matOut.row(matOut.rows()-1) = vec;

        }
    }

    return matOut;

}


void ImageGrabber::GrabImage(const ImageConstPtr& msgImage, const ImageConstPtr& msgDepth, 
            const geometry_msgs::PoseStampedConstPtr& msgOdom, const darknet_ros_msgs::BoundingBoxesConstPtr& msgBbox)
{
    // std::cout << "Get a Frame!!!" << std::endl;

    // std::cout << "OpenCV version: "
    //     << CV_MAJOR_VERSION << "." 
    //     << CV_MINOR_VERSION << "."
    //     << CV_SUBMINOR_VERSION
    //     << std::endl;

    cv::Mat imRGB, imDepth, imDepth32F;
    VectorXd pose, cam_pose;
    Vector4d bbox;

    double timestamp = msgImage->header.stamp.toSec();

    std::cout << std::endl << std::endl << std::endl;
    std::cout << "[ROS] Timestamp:" << timestamp << std::endl;

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvCopy(msgImage);
        if(!cv_ptrRGB->image.empty())
            imRGB = cv_ptrRGB->image.clone();
        else 
        {
            std::cout << "Empty RGB!" << std::endl;
            return;
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    cv_bridge::CvImageConstPtr cv_ptrDepth;
    try
    {
        cv_ptrDepth = cv_bridge::toCvCopy(msgDepth);
        if(!cv_ptrDepth->image.empty())
            imDepth = cv_ptrDepth->image.clone();
        else
        {
            std::cout << "Empty Depth!" << std::endl;
            return;
        }
        // imDepth32F.convertTo(imDepth, CV_16UC1, 1000);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // 获取里程计读数
    pose.resize(7, 1);
    pose << msgOdom->pose.position.x,msgOdom->pose.position.y,msgOdom->pose.position.z,
            msgOdom->pose.orientation.x,msgOdom->pose.orientation.y,msgOdom->pose.orientation.z,msgOdom->pose.orientation.w;

    // std::cout << "* Pose: " << pose.transpose() << std::endl;

    if( DatasetType == "Gazebo")
        cam_pose = odomToCamPose(pose); // 使用外参数获得相机位姿
    else
        cam_pose = pose;    // 保持原样.

    std::vector<darknet_ros_msgs::BoundingBox> boxes = msgBbox->bounding_boxes;
    // if( boxes.size() == 0 ) 
    // {
    //     std::cerr << "No detection. " << std::endl;
    //     return;
    // }
    
    // ------------- 正式处理 -----------------

    // 关键帧判断
    bool isKeyFrame = JudgeKeyFrame(cam_pose);

    // 所有帧都考虑
    isKeyFrame = true;

    if( isKeyFrame )
    {
        MatrixXd bboxMat = rosMsgToMat(boxes);        // bboxMat: id x1 y1 x2 y2 label rate [Instance]

        // 过滤部分label
        // std::set<int> vec_ignore{ 56, 72, 4 };
        std::set<int> vec_ignore;
        MatrixXd bboxMatFiltered = FilterDetectionMat(bboxMat, vec_ignore);
        // std::cout << " [ Attention ] Filter label: None." << std::endl;
        // std::cout << "bboxMatFiltered : " << std::endl << bboxMatFiltered << std::endl;
        mpSLAM->TrackWithObjects(timestamp, cam_pose, bboxMatFiltered, imDepth, imRGB, false);   // 其中 imRGB只用作可视化.

        // (double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd & bboxMat, const cv::Mat &imDepth, const cv::Mat &imRGB,
                    // bool withAssociation)
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