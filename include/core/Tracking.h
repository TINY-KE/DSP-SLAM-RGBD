#ifndef ELLIPSOIDSLAM_TRACKING_H
#define ELLIPSOIDSLAM_TRACKING_H

#include "System.h"
#include "FrameDrawer.h"
#include "Viewer.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Initializer.h"
#include "Optimizer.h"

#include <src/symmetry/Symmetry.h>
#include <src/pca/EllipsoidExtractor.h>
#include <src/plane/PlaneExtractor.h>
#include "DataAssociation.h"
#include <src/dense_builder/builder.h>

#include <src/plane/PlaneExtractorManhattan.h>
#include <src/Relationship/Relationship.h>

namespace EllipsoidSLAM{

class System;
class FrameDrawer;
class MapDrawer;
class Map;
class Viewer;
class Frame;
class Initializer;
class Optimizer;
class Symmetry;

enum OBJECT_MODEL
{
    POINT_MODEL = 0,
    QUADRIC_MODEL = 1
};

class Tracking {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
    Tracking(System* pSys, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             const string &strSettingPath);

    bool GrabPoseAndSingleObjectAnnotation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection);
    virtual bool GrabPoseAndObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB = cv::Mat(), bool withAssociation = false);
    bool GrabPoseAndObjects(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap, const cv::Mat &imDepth, const cv::Mat &imRGB = cv::Mat(), bool withAssociation = false);

    Frame* mCurrFrame;

    Eigen::Matrix3d mCalib;
    int mRows, mCols;

    void outputBboxMatWithAssociation();

    void SaveObjectHistory(const string& path);

    void OpenOptimization();
    void CloseOptimization();

    bool SavePointCloudMap(const string& path);

    std::vector<bool> checkKeyFrameForInstances(std::vector<int>& associations);

    // Single-frame Ellipsoid Extraction
    void OpenDepthEllipsoid();

    // Groundplane Estimation
    void OpenGroundPlaneEstimation();
    void CloseGroundPlaneEstimation();
    int GetGroundPlaneEstimationState();

    // Optimization with probabilistic data association
    bool NonparamOptimization(const OBJECT_MODEL& model = QUADRIC_MODEL);

    // Mannual Set Groundplane
    void SetGroundPlaneMannually(const Eigen::Vector4d &param);

    void LoadPointcloud(const string& strPcdDir, const string& strPointcloud_name, g2o::SE3Quat Ttrans=g2o::SE3Quat());
    std::vector<Frame*> GetAllFrames();

    // Save/load for debugging
    void Save(std::vector<Frame*>& pFrames);
    void Load(std::vector<Frame*>& pFrames);

    Builder* GetBuilder();

//private:
public:

    void UpdateObjectObservation(Frame* pFrame, bool withAssociation = false);

    void JudgeInitialization();

    bool isKeyFrameForVisualization(); 

    void ProcessVisualization();

    void RefreshObjectHistory();
    void ProcessGroundPlaneEstimation();
    void Update3DObservationDataAssociation(EllipsoidSLAM::Frame* pFrame, std::vector<int>& associations, std::vector<bool>& KeyFrameChecks);
    void UpdateDepthEllipsoidEstimation(EllipsoidSLAM::Frame* pFrame, bool withAssociation);
    void UpdateDepthEllipsoidUsingPointModel(EllipsoidSLAM::Frame* pFrame);

    std::vector<int> GetMannualAssociation(Eigen::MatrixXd &obsMat);

    void Update2DObservation(EllipsoidSLAM::Frame* pFrame);
    void GenerateObservationStructure(EllipsoidSLAM::Frame* pFrame);
    void InferObjectsWithSemanticPrior(EllipsoidSLAM::Frame* pFrame, bool use_input_pri, bool replace_detection);

    void TaskGroundPlane();
    void ActivateGroundPlane(g2o::plane &groundplane);
    void TaskManhattanPlanes(EllipsoidSLAM::Frame *pFrame);
    void VisualizeManhattanPlanes();

    void TaskRelationship(EllipsoidSLAM::Frame* pFrame);
    void RefineObjectsWithRelations(EllipsoidSLAM::Frame *pFrame);
    void ManageMemory();

    void AddNewFrame(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
        const cv::Mat &imDepth, const cv::Mat &imRGB);
    void SaveFrameAndDetectionResult(std::vector<Frame *> & vpFrames, bool save_cplanes = true);
    void SaveFrameAndPlaneResult(std::vector<Frame *> & vpFrames);
    void SaveGroundPlane();

    void TaskRelationshipWithPointModel(EllipsoidSLAM::Frame *pFrame);
    void UpdateObjectObservationWithPointModel(Frame* pFrame, bool withAssociation);
    void UpdateObjectObservationMultiSupportingPlanes(EllipsoidSLAM::Frame *pFrame, bool withAssociation);

    void ClearVisualization();

    bool DealWithOnline();

protected:

    g2o::ellipsoid* getObjectDataAssociation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection);

    System* mpSystem;
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    Initializer* mpInitializer;
    Map* mpMap;
    Optimizer* mpOptimizer;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    camera_intrinsic mCamera;

    std::vector<Frame*> mvpFrames;

    // Store observations in a map with instance id.
    // In the future, storing observations under Ellipsoid class separately would make it clearer.
    std::map<int, Observations> mmObjectObservations;

    Builder* mpBuilder;     // a dense pointcloud builder from visualization

    std::map<int, MatrixXd> mmObjectHistory;

    bool mbOpenOptimization;

    bool mbDepthEllipsoidOpened;
    EllipsoidExtractor* mpEllipsoidExtractor;
    std::map<int, Observation3Ds> mmObjectObservations3D;  // 3d observations indexed by instance ID

    DataAssociationSolver* pDASolver;

    int miGroundPlaneState; // 0: Closed  1: estimating 2: estimated 3: set by mannual
    g2o::plane mGroundPlane;
    PlaneExtractor* pPlaneExtractor;
    PlaneExtractorManhattan* pPlaneExtractorManhattan;
    int miMHPlanesState; // 0: Closed 1: estimating 2: estimated

    RelationExtractor* mpRelationExtractor;

    std::vector<g2o::SE3Quat> mvSavedFramePosesTwc;
};

}

#endif //ELLIPSOIDSLAM_TRACKING_H
