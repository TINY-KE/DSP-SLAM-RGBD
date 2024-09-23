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

#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>

#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>


/*
 * pub frames given.
 *
 */

using namespace std;

typedef vector<double> camOdom;

class OdomListener{

public:
    OdomListener(ros::NodeHandle* nh, vector<camOdom>* poses);
    void Callback(const sensor_msgs::ImageConstPtr& msg);

private:
    ros::NodeHandle* mpNh;
    vector<camOdom>* mvpPoses;
    ros::Publisher odom_pub;
    ros::Subscriber sub;
};

OdomListener::OdomListener(ros::NodeHandle *nh, vector<camOdom>* poses):mpNh(nh), mvpPoses(poses) {
    sub = mpNh->subscribe("/camera/rgb/image_raw", 1000, &OdomListener::Callback, this);

    odom_pub = nh->advertise<nav_msgs::Odometry>("/odom", 1000);

    cout << "初始化监听 /odom..." << endl;

}

vector<camOdom> readTruePoses(string loc){
    vector<camOdom> datas;
    ifstream fin(loc);
    string line;

    getline(fin, line); // 扔掉第一行数据

    while( getline(fin, line) )
    {
        camOdom data;
        vector<string> s;
        boost::split( s, line, boost::is_any_of( " " ), boost::token_compress_on );
        for (auto num : s) data.push_back(stod(num));
        datas.push_back(data);
    }

    return datas;
}

void publish_odom(ros::Publisher& odom_pub, tf::TransformBroadcaster& odom_broadcaster, camOdom data){
    //since all odometry is 6DOF we'll need a quaternion created from yaw
    geometry_msgs::Quaternion odom_quat;
    odom_quat.x = data[4];
    odom_quat.y = data[5];
    odom_quat.z = data[6];
    odom_quat.w = data[7];

    //next, we'll publish the odometry message over ROS
    nav_msgs::Odometry odom;
    odom.header.stamp.nsec = double(data[0])*1000000000;
    odom.header.frame_id = "world";

    //set the position
    odom.pose.pose.position.x = data[1];
    odom.pose.pose.position.y = data[2];
    odom.pose.pose.position.z = data[3];
    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "camera";
    odom.twist.twist.linear.x = 0;
    odom.twist.twist.linear.y = 0;
    odom.twist.twist.angular.z = 0;

    //publish the message
    odom_pub.publish(odom);
}

void OdomListener::Callback(const sensor_msgs::ImageConstPtr& msg)
{
    double timestamp_s = msg->header.stamp.toSec(); // 可自动做时间戳对齐
    ROS_INFO("Get /odom: [%f]", timestamp_s);
    static int i=0;

    for( auto p : *mvpPoses)
    {
        if( abs(p[0] - timestamp_s) < 0.01 )
        {
            // 找到了时间戳相同的数据.

            geometry_msgs::Quaternion odom_quat;
            odom_quat.x = p[4];
            odom_quat.y = p[5];
            odom_quat.z = p[6];
            odom_quat.w = p[7];

            //next, we'll publish the odometry message over ROS
            nav_msgs::Odometry odom;
            ros::Time time_now = ros::Time(p[0]);
            odom.header.stamp = time_now;
            odom.header.frame_id = "world";

            //set the position
            odom.pose.pose.position.x = p[1];
            odom.pose.pose.position.y = p[2];
            odom.pose.pose.position.z = p[3];
            odom.pose.pose.orientation = odom_quat;

            //set the velocity
            odom.child_frame_id = "camera";
            odom.twist.twist.linear.x = 0;
            odom.twist.twist.linear.y = 0;
            odom.twist.twist.angular.z = 0;

            //publish the message
            odom_pub.publish(odom);

            i++;
            ROS_INFO("-> [%d] Publish /odom_simu: [%f]", i, p[0]);
            break;
        }
    }

}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "odom_puber");
    ros::NodeHandle nh;
    ros::NodeHandle nph("~");
    // for true pose

    if( argc < 2)
        cout << "usage : path" << endl;

    ROS_INFO("Read poses from %s", argv[1]);
    // 读取相机真实姿态
    vector<camOdom> poses = readTruePoses(argv[1]);

    ROS_INFO("Get %d poses.", int(poses.size()));

    OdomListener listener(&nh, &poses);

    ROS_INFO("Wait for odom topic...");

    ros::spin();


}