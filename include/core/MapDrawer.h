#ifndef ELLIPSOIDSLAM_MAPDRAWER_H
#define ELLIPSOIDSLAM_MAPDRAWER_H

#include "Map.h"
#include <pangolin/pangolin.h>

#include <mutex>
#include <string>
#include <map>

using namespace std;

namespace EllipsoidSLAM{

class Map;

class MapDrawer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MapDrawer(const string &strSettingPath, Map* pMap);

    bool updateObjects();
    bool updateCameraState();


    bool drawObjects(double prob_thresh = 0);
    bool drawCameraState();
    bool drawGivenCameraState(g2o::SE3Quat* state, const Vector3d& color);

    bool drawEllipsoids(double prob_thresh = 0);
    bool drawObservationEllipsoids(double prob_thresh = 0);

    bool drawPlanes(int visual_group=0);

    bool drawPoints();

    void setCalib(Eigen::Matrix3d& calib);

    bool drawTrajectory();
    bool drawTrajectoryDetail();
    bool drawGivenTrajDetail(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color);

    bool drawTrajectoryWithName(const string& name);

    void SE3ToOpenGLCameraMatrix(g2o::SE3Quat &matIn, pangolin::OpenGlMatrix &M); // inverse matIn
    void SE3ToOpenGLCameraMatrixOrigin(g2o::SE3Quat &matIn, pangolin::OpenGlMatrix &M); // don't inverse matIn
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

    void drawPointCloudLists(); // draw all the point cloud lists 
    void drawPointCloudWithOptions(const std::map<std::string,bool> &options); // draw the point cloud lists with options opened

    void drawBoundingboxes();
    void drawConstrainPlanes(double prob_thresh = 0, int type = 0);

    void drawArrows();
    void drawLine(const Vector3d& start, const Vector3d& end, const Vector3d& color, double width, double alpha = 1.0);

    void SetTransformTge(const g2o::SE3Quat& Tge);

    void drawAxisNormal();

    void drawEllipsoidsLabelText(double prob_thresh, bool show_ellipsoids = true, bool show_observation = true);

private:

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;

    Map* mpMap;

    Eigen::Matrix3d mCalib;  

    void drawPlaneWithEquationDense(plane* p);
    void drawPlaneWithEquation(plane* p);
    void drawAllEllipsoidsInVector(std::vector<ellipsoid*>& ellipsoids);
    void drawLabelTextOfEllipsoids(std::vector<ellipsoid*>& ellipsoids);

    pangolin::OpenGlMatrix getGLMatrixFromCenterAndNormal(Vector3f& center, Vector3f& normal);

    void drawOneBoundingbox(Matrix3Xd& corners, Vector3d& color, double alpha = 1.0);

    bool drawGivenTrajWithColor(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color);
    bool drawGivenTrajWithColorLines(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color);

    void eigenMatToOpenGLMat(const Eigen::Matrix4d& matEigen, pangolin::OpenGlMatrix &M);

    g2o::SE3Quat mTge;  // estimation to groundtruth coordinite, for visualization only.
    bool mbOpenTransform;
};
}

#endif //ELLIPSOIDSLAM_MAPDRAWER_H
