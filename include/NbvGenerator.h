//
// Created by zhjd on 11/17/22.
//

#ifndef active_slam_vpsom_NBVGENERATOR_H
#define active_slam_vpsom_NBVGENERATOR_H

//ros
//#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>

//rrt
#include "rrt.h"
#include <ros/ros.h>

//eigen cv的convert
#include <Eigen/Dense>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/core/eigen.hpp>

//内部
#include "Map.h"
#include "MapObject.h"
#include "Converter.h"
#include "Tracking.h"
#include <random>

////movebase action
//#include <move_base_msgs/MoveBaseAction.h>
//#include <actionlib/client/simple_action_client.h>
//typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

#include "obstacles.h"


namespace ORB_SLAM2
{


class Tracking;
class FrameDrawer;
class MapPublisher;
class MapDrawer;
class System;

struct Candidate{
    cv::Mat pose;
    double reward;
//    BackgroudObject* bo;
};

struct localCandidate{
    double angle;
    double reward;
    int num;
};




class NbvGenerator {

public:
    NbvGenerator();
    NbvGenerator(Map* map, Tracking *pTracking, const string &strSettingPath);

    void Run();

    void RequestFinish();

private:
    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    bool mbStopped;
    bool mbStopRequested;
    std::mutex mMutexStop;

private:
    Map* mpMap;
    Tracking* mpTracker;

    ros::NodeHandle nh;

    int  mbPubGlobalGoal, mbPubLocalGoal;
    int mFakeLocalNum=0;
    int mbFakeBackgroudObjects;
    int mnObserveMaxNumBackgroudObject = 10;

    const char* CANDIDATE_NAMESPACE = "Candidate";
    const char* MAP_FRAME_ID = "map"; //  odom   imu_link   /ORB_SLAM/World    map
    float fCameraSize;
    float fPointSize;
    bool mbEnd_active_map = false;

//    ros::Publisher publisher_centroid;
//    ros::Publisher publisher_candidate;
    int mtest = 5;
//    ros::Publisher publisher_candidate_unsort;
//    ros::Publisher publisher_nbv;
//    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction>* mActionlib;
//    ros::Subscriber sub_reachgoal;
//    void SubReachGoal(const std_msgs::Bool::ConstPtr & msg);
    bool mbReachGoalFlag = false;

    vector<Candidate> mvGlobalCandidate;
    vector<localCandidate> mvLocalCandidate;
    Candidate NBV;
    vector<Candidate> mNBVs_old; //存储已到达的NBV，从而使下一个NBV尽量已到达的位置。
    double mNBVs_scale = 0;



    vector<Candidate> RotateCandidates(Candidate& initPose);
    double computeCosAngle_Signed(Eigen::Vector3d &v1,  Eigen::Vector3d &v2 , bool isSigned);

    void PublishGlobalNBVRviz(const vector<Candidate> &candidates);

//plane和背景物体的可视化
private:
    ros::Publisher pubCloud;
    ros::Publisher publisher_object_backgroud;
    geometry_msgs::Point corner_to_marker(Eigen::Vector3d& v);


private:
    float mfx, mfy, mcx, mcy;
    float mImageWidth, mImageHeight;
    float down_nbv_height;       //nbv的高度
    float mMaxPlaneHeight, mMinPlaneHeight, mMinPlaneSafeRadius, mGobalCandidateNum;
    cv::Mat mT_basefootprint_cam;       //相机在机器人底盘上的坐标
    //cv::Mat mT_world_initbaselink;     //初始机器人底盘在世界中的坐标
    cv::Mat mT_world_cam;     //初始相机在世界中的坐标
    double mNBV_Angle_correct;
    int mCandidate_num_topub; //展示的候选观测点的数量
    int mBackgroudObjectNum;    //背景物体的数量。先验
    double mPitch;  //相机的俯仰角
    //double mgreat_angle = 0;
    //std::mutex mMutexMamAngle;
    //double getMamGreadAngle();
    string mstrSettingPath;
    float mReward_dis, mReward_angle_cost;


//rrt
    enum rrt_status{
        running = true,
        success = false
    };

    bool status = running;

    void initializeMarkersParameters(visualization_msgs::Marker &sourcePoint,
                                     visualization_msgs::Marker &goalPoint,
                                     visualization_msgs::Marker &randomPoint,
                                     visualization_msgs::Marker &rrtTreeMarker,
                                     visualization_msgs::Marker &finalPath);

    vector< vector<geometry_msgs::Point> > getObstacles();


    void addBranchtoRRTTree(visualization_msgs::Marker &rrtTreeMarker, RRT::rrtNode &tempNode, RRT &myRRT);

    bool checkIfInsideBoundary(RRT::rrtNode &tempNode);

    bool checkIfOutsideObstacles3D(RRT::rrtNode &nearesetNode, RRT::rrtNode &tempNode, vector<MapObject*> obs);

    void generateTempPoint(RRT::rrtNode &tempNode);


    // 用于向 RRT (Rapidly-exploring Random Tree) 中添加新的节点 tempNode
    bool addNewPointtoRRT(RRT &myRRT, RRT::rrtNode &tempNode, double rrtStepSize, vector<MapObject*> obs );

    bool checkNodetoGoal(double X, double Y, double Z, RRT::rrtNode &tempNode);

    void setFinalPathData(vector< vector<double> > &rrtPaths, RRT &myRRT, int i, visualization_msgs::Marker &finalpath, double goalX, double goalY, double goalZ);

    void displayTheFinalPathNodeInfo(vector<double> path, RRT &myRRT)  ;

};

}


#endif //active_slam_vpsom_NBVGENERATOR_H
