#include <obstacles.h>
#include <geometry_msgs/Point.h>

using namespace ORB_SLAM2;

vector< vector<geometry_msgs::Point> > obstacles::getObstacleArray()
{
    vector<geometry_msgs::Point> obstaclePoint;
    geometry_msgs::Point point;
    // 1
    point.x = 50;
    point.y = 50;
    point.z = 50;

    obstaclePoint.push_back(point);

    // 2
    point.x = 50;
    point.y = 70;
    point.z = 50;

    obstaclePoint.push_back(point);

    // 3
    point.x = 70;
    point.y = 70;
    point.z = 50;

    obstaclePoint.push_back(point);

    // 4
    point.x = 70;
    point.y = 50;
    point.z = 50;
    obstaclePoint.push_back(point);

    // 5
    point.x = 50;
    point.y = 50;
    point.z = 70;
    obstaclePoint.push_back(point);

    // 6
    point.x = 50;
    point.y = 70;
    point.z = 70;
    obstaclePoint.push_back(point);

    // 7
    point.x = 70;
    point.y = 70;
    point.z = 70;
    obstaclePoint.push_back(point);

    // 8
    point.x = 70;
    point.y = 50;
    point.z = 70;
    obstaclePoint.push_back(point);

    // 9 和第一个点相同
    point.x = 50;
    point.y = 50;
    point.z = 50;
    obstaclePoint.push_back(point);

    obstacleArray.push_back(obstaclePoint);

    return obstacleArray;

}
