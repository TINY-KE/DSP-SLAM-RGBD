/**
* This file is part of https://github.com/JingwenWang95/DSP-SLAM
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

#ifndef MAPOBJECT_H
#define MAPOBJECT_H

#include <Eigen/Dense>
#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"
#include "Candidate.h"
//[zhjd]
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

namespace ORB_SLAM2 {

class KeyFrame;
class Map;
class Frame;
//class LocalMapping;

struct Cuboid3D{
        //     7------6
        //    /|     /|
        //   / |    / |
        //  4------5  |
        //  |  3---|--2
        //  | /    | /
        //  0------1
        // lenth ：corner_2[0] - corner_1[0]
        // width ：corner_2[1] - corner_3[1]
        // height：corner_2[2] - corner_6[2]

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        // 8 vertices.
        Eigen::Vector3f corner_1;
        Eigen::Vector3f corner_2;
        Eigen::Vector3f corner_3;
        Eigen::Vector3f corner_4;
        Eigen::Vector3f corner_5;
        Eigen::Vector3f corner_6;
        Eigen::Vector3f corner_7;
        Eigen::Vector3f corner_8;

        // 8 vertices (without rotation).
        Eigen::Vector3f corner_1_w;
        Eigen::Vector3f corner_2_w;
        Eigen::Vector3f corner_3_w;
        Eigen::Vector3f corner_4_w;
        Eigen::Vector3f corner_5_w;
        Eigen::Vector3f corner_6_w;
        Eigen::Vector3f corner_7_w;
        Eigen::Vector3f corner_8_w;

        Eigen::Vector3f cuboidCenter;       // the center of the Cube, not the center of mass of the object
        float x_min, x_max, y_min, y_max, z_min, z_max;     // the boundary in XYZ direction.

        float lenth;
        float width;
        float height;
        float mfRMax;      // 中心点与角点的最大半径

        //g2o::SE3Quat pose ;                               // 6 dof pose.
        cv::Mat pose_mat = cv::Mat::eye(4, 4, CV_32F);      //cv::mat形式的 物体在世界坐标系下的位姿
        //g2o::SE3Quat pose_without_yaw;                    // 6 dof pose without rotation.
        cv::Mat pose_noyaw_mat = cv::Mat::eye(4, 4, CV_32F);
        // angle.
        float rotY = 0.0;
        float rotP = 0.0;
        float rotR = 0.0;

        // line.
        float mfErrorParallel;
        float mfErroeYaw;
};


class MapObject {
public:
    MapObject(const Eigen::Matrix4f &T, const Eigen::Vector<float, 64> &vCode, const py::object& pyOptimizer, KeyFrame *pRefKF, Map *pMap );
    MapObject(const py::object& pyOptimizer, KeyFrame *pRefKF, Map *pMap);

    void AddObservation(KeyFrame *pKF, int idx);
    int Observations();
    std::map<KeyFrame*,size_t> GetObservations();
    void SetObjectPoseSim3(const Eigen::Matrix4f &Two);
    void SetObjectPoseSE3(const Eigen::Matrix4f &Two);
    void SetShapeCode(const Eigen::Vector<float, 64> &code);
    void UpdateReconstruction_foronlymono(const Eigen::Matrix4f &T, const Eigen::Vector<float, 64> &vCode);
    Eigen::Matrix4f GetPoseSim3();
    Eigen::Matrix4f GetPoseSE3();
    Eigen::Vector<float, 64> GetShapeCode();
    int GetIndexInKeyFrame(KeyFrame *pKF);
    void EraseObservation(KeyFrame *pKF);
    void SetBadFlag();
    bool isBad();
    void SetVelocity(const Eigen::Vector3f &v);
    void Replace(MapObject *pMO);
    bool IsInKeyFrame(KeyFrame *pKF);
    KeyFrame* GetReferenceKeyFrame();

    std::vector<MapPoint*> GetMapPointsOnObject_foronlymono();
    void AddMapPoints_foronlymono(MapPoint *pMP);
    void RemoveOutliersSimple();
    void RemoveOutliersModel();
    void ComputeCuboidPCA_onlyformono(bool updatePose);
    void EraseMapPoint(MapPoint *pMP);

    void SetRenderId(int id);
    int GetRenderId();
    void SetDynamicFlag();
    bool isDynamic();

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix4f SE3Two;
    Eigen::Matrix4f SE3Tow;
    Eigen::Matrix4f Sim3Two;
    Eigen::Matrix4f Sim3Tow;
    Eigen::Matrix3f Rwo;
    Eigen::Vector3f two;
    float scale;
    float invScale;
    Eigen::Vector<float, 64> vShapeCode;

    // Keyframes observing the point and associated index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Reference KeyFrame
    KeyFrame *mpRefKF;
    KeyFrame *mpNewestKF;
    long unsigned int mnBALocalForKF;
    long unsigned int mnAssoRefID;
    long unsigned int mnFirstKFid;

    // variables used for loop closing
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    long unsigned int mnLoopObjectForKF;
    long unsigned int mnBAGlobalForKF;
    MapObject *mpReplaced;
    Eigen::Matrix4f mTwoGBA;

    bool reconstructed;
    std::set<MapPoint*> map_points_foronlymono;

    // cuboid
    float w;
    float h;
    float l;

    // Bad flag (we do not currently erase MapObject from memory)
    bool mbBad;
    bool mbDynamic;
    Eigen::Vector3f velocity;
    Map *mpMap;

    int nObs;
    static int nNextId;
    int mnId; // Object ID
    int mRenderId; // Object ID in the renderer
    Eigen::MatrixXf vertices;
    Eigen::MatrixXi faces;

    std::mutex mMutexObject;
    std::mutex mMutexFeatures;

    static bool lId(MapObject* pMO1, MapObject* pMO2){
        return pMO1->mnId < pMO2->mnId;
    }

public:
    py::object pyOptimizer;

    double mdSdfLoss;
    void compute_sdf_loss_of_all_inside_points();
    double compute_sdf_loss(double x, double y, double z);

    void compute_NBV();
    NBV* nbv;
    Cuboid3D mCuboid3D;
    void updateCuboid3D();
};

}
#endif //MAPOBJECT_H
