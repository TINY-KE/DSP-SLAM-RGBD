#ifndef ELLIPSOIDSLAM_MAP_H
#define ELLIPSOIDSLAM_MAP_H

#include "Ellipsoid.h"
#include "Geometry.h"
#include "Plane.h"
#include <mutex>
#include <set>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>	

using namespace g2o;

namespace EllipsoidSLAM
{
    class SE3QuatWithStamp
    {
    public:
        g2o::SE3Quat pose;
        double timestamp;
    };
    typedef std::vector<SE3QuatWithStamp*> Trajectory;

    class Boundingbox
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        Vector3d color;
        double alpha;
        Matrix3Xd points;
    };

    class Arrow
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Vector3d center;
        Vector3d norm;
        Vector3d color;    
    };

    enum ADD_POINT_CLOUD_TYPE
    {
        REPLACE_POINT_CLOUD = 0,
        ADD_POINT_CLOUD = 1
    };

    enum DELETE_POINT_CLOUD_TYPE
    {
        COMPLETE_MATCHING = 0,
        PARTIAL_MATCHING = 1
    };

    class Map
    {
    public:
        Map();

        void addEllipsoid(ellipsoid* pObj);
        std::vector<ellipsoid*> GetAllEllipsoids();

        void addPlane(plane* pPlane, int visual_group = 0);
        std::vector<plane*> GetAllPlanes();
        void clearPlanes();

        void setCameraState(g2o::SE3Quat* state);
        g2o::SE3Quat* getCameraState();

        void addCameraStateToTrajectory(g2o::SE3Quat* state);
        std::vector<g2o::SE3Quat*> getCameraStateTrajectory();
        void ClearCameraTrajectory();

        void addToTrajectoryWithName(SE3QuatWithStamp* state, const string& name);
        Trajectory getTrajectoryWithName(const string& name);
        bool clearTrajectoryWithName(const string& name);
        bool addOneTrajectory(Trajectory& traj, const string& name);

        void addPoint(PointXYZRGB* pPoint);
        void addPointCloud(PointCloud* pPointCloud);
        void clearPointCloud();
        std::vector<PointXYZRGB*> GetAllPoints();

        std::vector<ellipsoid*> getEllipsoidsUsingLabel(int label);

        std::map<int, ellipsoid*> GetAllEllipsoidsMap();

        bool AddPointCloudList(const string& name, PointCloud* pCloud, int type = 0);   // type 0: replace when exist,  type 1: add when exist
        bool DeletePointCloudList(const string& name, int type = 0);    // type 0: complete matching, 1: partial matching
        bool ClearPointCloudLists();

        // 针对新的接口
        bool AddPointCloudList(const string& name, std::vector<pcl::PointCloud<pcl::PointXYZRGB>>& vCloudPCL, g2o::SE3Quat& Twc, int type = REPLACE_POINT_CLOUD);

        std::map<string, PointCloud*> GetPointCloudList();
        PointCloud GetPointCloudInList(const string& name);

        void addArrow(const Vector3d& center, const Vector3d& norm, const Vector3d& color);
        std::vector<Arrow> GetArrows();
        void clearArrows();

    protected:
        std::vector<ellipsoid*> mspEllipsoids;
        std::set<plane*> mspPlanes;

        std::mutex mMutexMap;

        g2o::SE3Quat* mCameraState;   // Twc
        std::vector<g2o::SE3Quat*> mvCameraStates;      // Twc  camera in world
        std::map<string, Trajectory> mmNameToTrajectory;

        std::set<PointXYZRGB*> mspPoints;  
        std::map<string, PointCloud*> mmPointCloudLists; // name-> pClouds

        std::vector<Arrow> mvArrows;
    public:
        // those visual ellipsoids are for visualization only and DO NOT join the optimization
        void addEllipsoidVisual(ellipsoid* pObj);
        std::vector<ellipsoid*> GetAllEllipsoidsVisual();
        void ClearEllipsoidsVisual();

        void addEllipsoidObservation(ellipsoid* pObj);
        std::vector<ellipsoid*> GetObservationEllipsoids();
        void ClearEllipsoidsObservation();

        // interface for visualizing boungding box
        void addBoundingbox(Boundingbox* pBox);
        std::vector<Boundingbox*> GetBoundingboxes();
        void ClearBoundingboxes();

    protected:
        std::vector<ellipsoid*> mspEllipsoidsVisual;
        std::vector<ellipsoid*> mspEllipsoidsObservation;
        std::vector<Boundingbox*> mvBoundingboxes;

    };
}

#endif //ELLIPSOIDSLAM_MAP_H
