//
// Created by zhjd on 11/17/22.
//

#include "NbvGenerator.h"

//NBV
#include <std_msgs/Float64.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>

namespace ORB_SLAM2
{





NbvGenerator::NbvGenerator(){}

NbvGenerator::NbvGenerator(Map* map, Tracking *pTracking, const string &strSettingPath):
mpMap(map), mpTracker(pTracking)
{

}

void NbvGenerator::Run() {

    ros::Publisher rrt_publisher = nh.advertise<visualization_msgs::Marker>("path_planner_rrt",1);

    visualization_msgs::Marker sourcePoint;
    visualization_msgs::Marker goalPoint;
    visualization_msgs::Marker randomPoint;
    visualization_msgs::Marker rrtTreeMarker;
    visualization_msgs::Marker finalPath;

    initializeMarkersParameters(sourcePoint, goalPoint, randomPoint, rrtTreeMarker, finalPath);

    while(ros::ok() ){


        // 获取所有物体
        auto mvpMapObjects = mpMap->GetAllMapObjects();


        if(mvpMapObjects.empty())
            continue;

        //  开始的位置
        double startX, startY, startZ;
        startX = startY = startZ = 0.0;

        sourcePoint.pose.position.x = startX;
        sourcePoint.pose.position.y = startY;
        sourcePoint.pose.position.z = startZ;

        //   选择最佳的物体 -- 目标位置
        auto bestob = mvpMapObjects.front();
        if (!bestob)
            continue;
        // We only consider the object in the middle
        if (bestob->mnId != 0)
            continue;

        //bestob->compute_NBV();
        //bestob->compute_sdf_loss_of_all_inside_points();
        //std::cout<<"sdf shape 可用"<<std::endl;

        bestob->compute_NBV();
        auto NBV = bestob->nbv;
        if(!NBV)
            continue;

        double   goalX, goalY, goalZ;
        std::cout<<"NBV->centor_x:"<<NBV->centor_x<<std::endl;
        goalX = NBV->centor_x;
        goalY = NBV->centor_y;
        goalZ = NBV->centor_z;
        goalPoint.pose.position.x = goalX;
        goalPoint.pose.position.y = goalY;
        goalPoint.pose.position.z = goalZ;

        rrt_publisher.publish(sourcePoint);
        rrt_publisher.publish(goalPoint);
        ros::spinOnce();
        ros::Duration(0.01).sleep();

        srand (time(NULL));

        RRT myRRT(startX, startY, startZ);

        //步长
        double rrtStepSize = 0.1;  //每次在树中扩展时移动的步长大小

        vector< vector<double> > rrtPaths;

        vector<double> path;
        int rrtPathLimit = 1;

        int shortestPathLength = 9999;
        int shortestPath = -1;

        RRT::rrtNode tempNode;

        vector< vector<geometry_msgs::Point> >  obstacleList = getObstacles();


        bool addNodeResult = false, nodeToGoal = false;


        // rrt算法
        status = running;
        while( status)  //ros::ok() &&
        {
            if(rrtPaths.size() < rrtPathLimit)
            {

                generateTempPoint(tempNode);

                addNodeResult = addNewPointtoRRT(myRRT,tempNode,rrtStepSize,mvpMapObjects);

                if(addNodeResult)
                {
                    // 用于绘图
                    addBranchtoRRTTree(rrtTreeMarker,tempNode,myRRT);

                    nodeToGoal = checkNodetoGoal(goalX, goalY, goalZ, tempNode);

                    if(nodeToGoal)
                    {
                        path = myRRT.getRootToEndPath(tempNode.nodeID);
                        displayTheFinalPathNodeInfo(path, myRRT);
                        rrtPaths.push_back(path);
                        std::cout<<"New Path Found. Total paths "<<rrtPaths.size()<<endl;
                    }
                }
            }
            else
            {
                status = success;
                std::cout<<"Finding Optimal Path"<<endl;
                for(int i=0; i<rrtPaths.size();i++)
                {
                    if(rrtPaths[i].size() < shortestPath)
                        // 如果当前路径长度小于最短路径长度 shortestPath，则执行以下代码块。
                    {
                        shortestPath = i;
                        shortestPathLength = rrtPaths[i].size();
                    }
                }
                // 用于可视化。 类型为visualization_msgs::Marker &finalpath,
                setFinalPathData(rrtPaths, myRRT, shortestPath, finalPath, goalX, goalY, goalZ);
                rrt_publisher.publish(finalPath);
            }
            rrt_publisher.publish(sourcePoint);
            rrt_publisher.publish(goalPoint);
            rrt_publisher.publish(rrtTreeMarker);
            ros::spinOnce();
            ros::Duration(0.0005).sleep();
        }
    }
}


void NbvGenerator::initializeMarkersParameters(visualization_msgs::Marker &sourcePoint,
                                               visualization_msgs::Marker &goalPoint,
                                               visualization_msgs::Marker &randomPoint,
                                               visualization_msgs::Marker &rrtTreeMarker,
                                               visualization_msgs::Marker &finalPath)
{

    sourcePoint.header.frame_id    = goalPoint.header.frame_id    = randomPoint.header.frame_id    = rrtTreeMarker.header.frame_id    = finalPath.header.frame_id    = "map";
    sourcePoint.header.stamp       = goalPoint.header.stamp       = randomPoint.header.stamp       = rrtTreeMarker.header.stamp       = finalPath.header.stamp       = ros::Time::now();
    sourcePoint.ns                 = goalPoint.ns                 = randomPoint.ns                 = rrtTreeMarker.ns                 = finalPath.ns                 = "path_planner";
    sourcePoint.action             = goalPoint.action             = randomPoint.action             = rrtTreeMarker.action             = finalPath.action             = visualization_msgs::Marker::ADD;
    sourcePoint.pose.orientation.w = goalPoint.pose.orientation.w = randomPoint.pose.orientation.w = rrtTreeMarker.pose.orientation.w = finalPath.pose.orientation.w = 1.0;


    sourcePoint.id    = 0;
    goalPoint.id      = 1;
    randomPoint.id    = 2;
    rrtTreeMarker.id  = 3;
    finalPath.id      = 4;


    rrtTreeMarker.type                                    = visualization_msgs::Marker::LINE_LIST;
    finalPath.type                                        = visualization_msgs::Marker::LINE_STRIP;
    sourcePoint.type  = goalPoint.type = randomPoint.type = visualization_msgs::Marker::SPHERE;

    double scale = 30;

    rrtTreeMarker.scale.x = 0.2/scale;
    finalPath.scale.x     = 1/scale;

    sourcePoint.scale.x   = goalPoint.scale.x = 2/scale;
    sourcePoint.scale.y   = goalPoint.scale.y = 2/scale;
    sourcePoint.scale.z   = goalPoint.scale.z = 1/scale;

    randomPoint.scale.x = 2/scale;
    randomPoint.scale.y = 2/scale;
    randomPoint.scale.z = 1/scale;

    sourcePoint.color.r   = 1.0f;
    goalPoint.color.g     = 1.0f;

    randomPoint.color.b   = 1.0f;

    rrtTreeMarker.color.r = 0.8f;
    rrtTreeMarker.color.g = 0.4f;

    finalPath.color.r = 0.2f;
    finalPath.color.g = 0.2f;
    finalPath.color.b = 1.0f;

    sourcePoint.color.a = goalPoint.color.a = randomPoint.color.a = rrtTreeMarker.color.a = finalPath.color.a = 1.0f;
}

vector< vector<geometry_msgs::Point> > NbvGenerator::getObstacles()
{
    obstacles obst;
    return obst.getObstacleArray();
}


void NbvGenerator::addBranchtoRRTTree(visualization_msgs::Marker &rrtTreeMarker, RRT::rrtNode &tempNode, RRT &myRRT)
{

    geometry_msgs::Point point;

    point.x = tempNode.posX;
    point.y = tempNode.posY;
    point.z = tempNode.posZ;
    rrtTreeMarker.points.push_back(point);


    RRT::rrtNode parentNode = myRRT.getParent(tempNode.nodeID);

    point.x = parentNode.posX;
    point.y = parentNode.posY;
    point.z = parentNode.posZ;

    rrtTreeMarker.points.push_back(point);

}

bool NbvGenerator::checkIfInsideBoundary(RRT::rrtNode &tempNode)
{
    // if(tempNode.posX < 0 || tempNode.posY < 0  || tempNode.posZ < 0 || tempNode.posX > 100 || tempNode.posY > 100 || tempNode.posZ > 100) return false;
    // else return true;

    // 在自己的场景中，没有边界。所以都是true
    return true;
}



bool NbvGenerator::checkIfOutsideObstacles3D(RRT::rrtNode &nearesetNode, RRT::rrtNode &tempNode, const vector<MapObject*> obs)
{
    //1. 建立以nearesetNode和tempNode连线为对角线的立方体
    double length = abs((nearesetNode.posX - tempNode.posX));
    double width = abs((nearesetNode.posY - tempNode.posY));
    double heigth = abs((nearesetNode.posZ - tempNode.posZ));

    // 中间点
    RRT::rrtNode bodyCenter;
    bodyCenter.posX = (nearesetNode.posX + tempNode.posX)/2;
    bodyCenter.posY = (nearesetNode.posY + tempNode.posY)/2;
    bodyCenter.posZ = (nearesetNode.posZ + tempNode.posZ)/2;

    vector<geometry_msgs::Point> vertices;
    geometry_msgs::Point vertice;

    vertice.x = bodyCenter.posX - length/2;
    vertice.y = bodyCenter.posY - width/2;
    vertice.z = bodyCenter.posZ - heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX - length/2;
    vertice.y = bodyCenter.posY + width/2;
    vertice.z = bodyCenter.posZ - heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX + length/2;
    vertice.y = bodyCenter.posY + width/2;
    vertice.z = bodyCenter.posZ - heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX + length/2;
    vertice.y = bodyCenter.posY - width/2;
    vertice.z = bodyCenter.posZ - heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX - length/2;
    vertice.y = bodyCenter.posY - width/2;
    vertice.z = bodyCenter.posZ + heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX - length/2;
    vertice.y = bodyCenter.posY + width/2;
    vertice.z = bodyCenter.posZ + heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX + length/2;
    vertice.y = bodyCenter.posY + width/2;
    vertice.z = bodyCenter.posZ + heigth/2;
    vertices.push_back(vertice);


    vertice.x = bodyCenter.posX + length/2;
    vertice.y = bodyCenter.posY - width/2;
    vertice.z = bodyCenter.posZ + heigth/2;
    vertices.push_back(vertice);

    //2.判断立方体的八个顶点，是否在障碍物内。
    for(int i=0; i < vertices.size(); i++)
    {
        for(auto ob: obs){
            //if(ob->compute_sdf_loss(vertices[i].x,vertices[i].y,vertices[i].z) <= 0.0)
            //    return false;
        }
    }
//    ROS_INFO("NOT INSIDE OF OBJECTS");
    return true;
}

void NbvGenerator::generateTempPoint(RRT::rrtNode &tempNode)
{
    // 1
    //std::random_device rd;  // 随机设备作为种子
    //std::mt19937 gen(rd()); // Mersenne Twister 19937 作为随机数引擎
    //std::uniform_real_distribution<double> dis(0.01, 0.1); // 均匀分布
    //
    //double x = dis(gen); // 生成随机数
    //double y = dis(gen); // 生成随机数
    //double z = dis(gen); // 生成随机数

    int x = rand() % 150 + 1;
    int y = rand() % 150 + 1;
    int z = rand() % 150 + 1;

    tempNode.posX = x;
    tempNode.posY = y;
    tempNode.posZ = z;
}

// 用于向 RRT (Rapidly-exploring Random Tree) 中添加新的节点 tempNode
bool NbvGenerator::addNewPointtoRRT(RRT &myRRT, RRT::rrtNode &tempNode, double rrtStepSize, const vector<MapObject*> obs )
{
    int nearestNodeID = myRRT.getNearestNodeID(tempNode.posX, tempNode.posY, tempNode.posZ);  //找到最近节点，作为父节点。

    RRT::rrtNode nearestNode = myRRT.getNode(nearestNodeID);

    // 生成新节点tempNode
    double theta1 = atan2(tempNode.posZ - nearestNode.posZ, sqrt(pow(tempNode.posX - nearestNode.posX, 2) + pow(tempNode.posY - nearestNode.posY, 2) ));
    double theta2 = atan2(sqrt(pow(tempNode.posY - nearestNode.posY, 2) + pow(tempNode.posZ - nearestNode.posZ, 2)), tempNode.posX - nearestNode.posX);
    double theta3 = atan2(sqrt(pow(tempNode.posZ - nearestNode.posZ, 2) + pow(tempNode.posX - nearestNode.posX, 2)), tempNode.posY - nearestNode.posY);
    tempNode.posZ = nearestNode.posZ + (1 * sin(theta1));
    tempNode.posY = nearestNode.posY + (rrtStepSize * cos(theta3));
    tempNode.posX = nearestNode.posX + (rrtStepSize * cos(theta2));

    if(checkIfInsideBoundary(tempNode) && checkIfOutsideObstacles3D(nearestNode, tempNode, obs))
    {

        tempNode.parentID = nearestNodeID;
        tempNode.nodeID = myRRT.getTreeSize();
        myRRT.addNewNode(tempNode);
        return true;
    }
    else
        return false;
}


bool NbvGenerator::checkNodetoGoal(double X, double Y, double Z, RRT::rrtNode &tempNode)
{
    double distance = sqrt(pow(X-tempNode.posX,2)+pow(Y-tempNode.posY,2)+pow(Z-tempNode.posZ,2));
    if(distance < 10)
    {
        return true;
    }
    return false;
}

void NbvGenerator::setFinalPathData(vector< vector<double> > &rrtPaths, RRT &myRRT, int i, visualization_msgs::Marker &finalpath, double goalX, double goalY, double goalZ)
{
    RRT::rrtNode tempNode;
    geometry_msgs::Point point;
    for(int j=0; j<rrtPaths[i].size();j++)
    {
        tempNode = myRRT.getNode(rrtPaths[i][j]);
        point.x = tempNode.posX;
        point.y = tempNode.posY;
        point.z = tempNode.posZ;

        finalpath.points.push_back(point);
    }

    point.x = goalX;
    point.y = goalY;
    point.z = goalZ;
    finalpath.points.push_back(point);
}
void NbvGenerator::displayTheFinalPathNodeInfo(vector<double> path, RRT &myRRT)
{
    for(int i=0; i<path.size(); i++)
    {
        RRT::rrtNode testNode = myRRT.getNode(path[i]);
        std::cout << "path[i] = " << path[i] << std::endl;
        std::cout << "testNode.nodeID = " << testNode.nodeID << std::endl;
        std::cout << "testNode.parentID = " << testNode.parentID << std::endl;
        std::cout << "testNode.posX = " << testNode.posX << std::endl;
        std::cout << "testNode.posY = " << testNode.posY << std::endl;
        std::cout << "testNode.posZ = " << testNode.posZ << std::endl;
        std::cout << "------------" << std::endl;
    }
}


























vector<Candidate>  NbvGenerator::RotateCandidates(Candidate& initPose){
    vector<Candidate> cands;
    cv::Mat T_w_cam = initPose.pose;   //初始的相机在世界的坐标
    cv::Mat T_w_body = cv::Mat::eye(4, 4, CV_32F); T_w_body = T_w_cam * mT_basefootprint_cam.inv() ;  //初始的机器人底盘在世界的坐标
    cv::Mat T_w_body_new;    //旋转之后的机器人位姿
    int mDivide = 36;
    for(int i=0; i<=mDivide; i++){
            double angle = M_PI/mDivide * i - M_PI/2.0 ;
            //旋转
            Eigen::AngleAxisd rotation_vector (angle, Eigen::Vector3d(0,0,1));
            Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();  //分别加45度
            //Eigen::Isometry3d trans_matrix;
            //trans_matrix.rotate(rotation_vector);
            //trans_matrix.pretranslate(Vector3d(0,0,0));
            cv::Mat rotate_mat = Converter::toCvMat(rotation_matrix);

            //平移
            cv::Mat t_mat = (cv::Mat_<float>(3, 1) << 0, 0, 0);

            //总变换矩阵
            cv::Mat trans_mat = cv::Mat::eye(4, 4, CV_32F);
            rotate_mat.copyTo(trans_mat.rowRange(0, 3).colRange(0, 3));
            t_mat.copyTo(trans_mat.rowRange(0, 3).col(3));

            T_w_body_new = T_w_body * trans_mat;   //旋转之后的机器人位姿
            T_w_cam = T_w_body_new * mT_basefootprint_cam;   //旋转之后的相机位姿
            Candidate temp;
            temp.pose = T_w_cam;
            cands.push_back(temp);
        }
    return  cands;
}

geometry_msgs::Point NbvGenerator::corner_to_marker(Eigen::Vector3d& v){
    geometry_msgs::Point point;
    point.x = v[0];
    point.y = v[1];
    point.z = v[2];
    return point;
}


void NbvGenerator::PublishGlobalNBVRviz(const vector<Candidate> &candidates)
{
    if(candidates.empty())
        return ;

    // color.
    std::vector<vector<float> > colors_bgr{ {255,0,0},  {255,125,0},  {255,255,0 },  {0,255,0 },    {0,0,255},  {0,255,255},  {255,0,255},  {0,0,0}    };



    for(int i=0; i < candidates.size()  /*mCandidate_num_topub*/; i++)
    {
        //cv::Mat Tcw = Tcws[i];
        vector<float> color ;
        //if(i<7)
        //    color = colors_bgr[i];
        //else
        //    color = colors_bgr[7];

        float d = 0.1;
        //Camera is a pyramid. Define in camera coordinate system
        cv::Mat o = (cv::Mat_<float>(4, 1) << 0, 0, 0, 1);
        cv::Mat p1 = (cv::Mat_<float>(4, 1) << d, d * 0.8, d * 0.5, 1);
        cv::Mat p2 = (cv::Mat_<float>(4, 1) << d, -d * 0.8, d * 0.5, 1);
        cv::Mat p3 = (cv::Mat_<float>(4, 1) << -d, -d * 0.8, d * 0.5, 1);
        cv::Mat p4 = (cv::Mat_<float>(4, 1) << -d, d * 0.8, d * 0.5, 1);

        cv::Mat Twc = candidates[i].pose.clone();//Tcw.inv();
        cv::Mat ow = Twc * o;
        cv::Mat p1w = Twc * p1;
        cv::Mat p2w = Twc * p2;
        cv::Mat p3w = Twc * p3;
        cv::Mat p4w = Twc * p4;



        //ROS_INFO("Candidate %d, x:%f, y:%f, reward:%f", i, Twc.at<float>(0,3), Twc.at<float>(1,3), candidates[i].reward);
    }
}


double NbvGenerator::computeCosAngle_Signed( Eigen::Vector3d &v1 /*基坐标轴*/,  Eigen::Vector3d &v2, bool isSigned/*为1时，角度范围为360度*/){
    //Eigen::Vector3d view(   objectPose.at<float>(0,3)-candidate_pose.at<float>(0,3),
    //                        objectPose.at<float>(1,3)-candidate_pose.at<float>(1,3),
    //                        0.0     );
    //double cosTheta  = view.dot(ie) / (view.norm() * ie.norm()); //角度cos值
    //return (cosTheta); //std::acos(cosTheta)
    double cosValNew = v1.dot(v2) / (v1.norm()*v2.norm()); //通过向量的点乘, 计算角度cos值
    double angleNew = acos(cosValNew) * 180 / M_PI;     //弧度角

    // 如果为360度内的角度，则以v1是基坐标轴，如果v2逆时针旋转到zerovec，则修正夹角
    if(isSigned){
        if (v2.cross(v1)(2) > 0) {
            angleNew = 360 - angleNew;
        }
    }

    return angleNew;
}



};

