// Update: 2020-1-7
// 根据 rgb 图像发布 bbox

//
// Created by liaoziwei on 18-11-5.
//

// 本文件用于发布校准坐标系统之后的里程计数据.
// 根据话题发布odom消息的时间戳来发布相应的sift消息.

/*
 * 19-2-22 Update:
 * 基于旧版本函数针对Kitti数据集来做里程计的发布模拟。
 *
 * 11-24 Update:
 * 将该函数做一定的通用性。即一个坐标校准函数来适应ORB的数据来源。
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <vector>

#include <fstream>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <include/utils/dataprocess_utils.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>

/*
 * pub frames given.
 *
 */

using namespace std;
using namespace Eigen;


class ImageListener{

public:
    ImageListener(ros::NodeHandle* nh, const string& dataset_dir);
    void Callback(const sensor_msgs::ImageConstPtr& msg);

    void publishBboxes(MatrixXd &mat);

private:
    ros::NodeHandle* mpNh;
    ros::Publisher pub_bboxes;
    ros::Subscriber sub;

    string msDatasetDir;
};

ImageListener::ImageListener(ros::NodeHandle *nh, const string& dataset_dir):mpNh(nh) {
    msDatasetDir = dataset_dir;
    sub = mpNh->subscribe("/camera/rgb/image_raw", 1000, &ImageListener::Callback, this);
    pub_bboxes = mpNh->advertise<darknet_ros_msgs::BoundingBoxes>("/darknet_ros/bounding_boxes", 10, false);
    // bbox_pub = nh->advertise<nav_msgs::Odometry>("odom_simu", 1000);

    cout << "初始化监听 /camera/rgb/raw_image..." << endl;

}

void ImageListener::Callback(const sensor_msgs::ImageConstPtr& msg)
{
    double time_stamp = msg->header.stamp.toSec();

    string fileName = msDatasetDir + to_string(time_stamp) + ".txt";
    MatrixXd bboxesMat = readDataFromFile(fileName.c_str());

    std::cout << "publish bboxes from " << fileName << std::endl;

    // 构建数据结构.
    darknet_ros_msgs::BoundingBoxes bboxes;
    int row_num = bboxesMat.rows();
    for( int i=0;i<row_num; i++)
    {
        VectorXd vec = bboxesMat.row(i);
        darknet_ros_msgs::BoundingBox boundingBox;
        
        
        boundingBox.xmin = vec[1];// row : id x1 y1 x2 y2 label rate 
        boundingBox.ymin = vec[2];
        boundingBox.xmax = vec[3];
        boundingBox.ymax = vec[4];
        boundingBox.id = vec(5);
        boundingBox.probability = vec[6];

        boundingBox.Class = "unDef";

        bboxes.bounding_boxes.push_back(boundingBox);
    }

    // 发布.
    bboxes.header.stamp = msg->header.stamp;
    bboxes.header.frame_id = "detection";
    //bboxes.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    pub_bboxes.publish(bboxes);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "pub_bbox");
    ros::NodeHandle nh;
    ros::NodeHandle nph("~");
    // for true pose

    if( argc < 2)
        cout << "usage : path_dataset" << endl;

    ImageListener listener(&nh, argv[1]);

    ROS_INFO("Wait for image topic...");

    ros::spin();


}