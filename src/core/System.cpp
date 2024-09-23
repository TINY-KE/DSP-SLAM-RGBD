#include "include/core/System.h"
#include "include/core/Map.h"
#include "include/core/Frame.h"
#include "include/core/MapDrawer.h"
#include "include/core/Viewer.h"
#include "include/core/Tracking.h"
#include "include/core/FrameDrawer.h"

#include "include/utils/dataprocess_utils.h"

#include "src/config/Config.h"

EllipsoidSLAM::Map* expMap;

namespace EllipsoidSLAM
{

    System::System(const string &strSettingsFile, const bool bUseViewer) {

        cout << endl <<
        "EllipsoidSLAM Project 2019, Beihang University." << endl;

        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            cerr << " * State : " << fsSettings.state << endl;
            exit(-1);
        }
        
        // Initialize global settings.
        Config::Init();
        Config::SetParameterFile(strSettingsFile);

        bool use_infer_detection = Config::Get<int>("System.MonocularInfer.Open") > 0;
        cout << " Input Sensor: ";
        if(use_infer_detection)
            cout << "Monocular" << endl;
        else 
            cout << "RGB-D" << endl;

        //Create the Map
        mpMap = new Map();

        mpFrameDrawer = new FrameDrawer(mpMap);

        //Create Drawers. These are used by the Viewer
        mpMapDrawer = new MapDrawer(strSettingsFile, mpMap);

        mpTracker = new Tracking(this, mpFrameDrawer, mpMapDrawer, mpMap, strSettingsFile);
        SetTracker(mpTracker);
        
        //Initialize the Viewer thread and launch
        if(bUseViewer)
        {
            mpViewer = new Viewer(this, strSettingsFile, mpMapDrawer);
            mptViewer = new thread(&Viewer::run, mpViewer);
            mpViewer->SetFrameDrawer(mpFrameDrawer);
        }

        OpenDepthEllipsoid();   // Open Single-Frame Ellipsoid Extraction
        mpTracker->OpenGroundPlaneEstimation();     // Open Groundplane Estimation.

        expMap = mpMap;
    }

    bool System::TrackWithObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd & bboxMat, const cv::Mat &imDepth, const cv::Mat &imRGB,
                    bool withAssociation)
    {
        return mpTracker->GrabPoseAndObjects(timestamp, pose, bboxMat, imDepth, imRGB, withAssociation);
    }

    Map* System::getMap() {
        return mpMap;
    }

    MapDrawer* System::getMapDrawer() {
        return mpMapDrawer;
    }

    FrameDrawer* System::getFrameDrawer() {
        return mpFrameDrawer;
    }

    Viewer* System::getViewer() {
        return mpViewer;
    }

    Tracking* System::getTracker() {
        return mpTracker;
    }

    MatrixXd System::GetObjects()
    {
        auto ellipsoids = mpMap->GetAllEllipsoidsVisual();
        
        MatrixXd objMat;objMat.resize(0, 11);
        int valid_objs = 0;

        double config_prob_thresh = Config::ReadValue<double>("Dynamic.Optimizer.EllipsoidProbThresh");
        std::cout << " ==== Objects Threshold : " << config_prob_thresh << " ==== " << std::endl;

        for(auto e : ellipsoids)
        {
            if(e->prob < config_prob_thresh) continue;

            Vector9d vec = e->toMinimalVector();
            VectorXd vec_instance; vec_instance.resize(11);
            vec_instance << e->miInstanceID, vec, e->miLabel;

            addVecToMatirx(objMat, vec_instance);
            valid_objs++;
        }

        return objMat;
    }

    void System::SaveObjectsToFile(string &path){
        MatrixXd objMat = GetObjects();
        saveMatToFile(objMat, path.c_str());

        std::cout << "Save " << objMat.rows() << " VISUAL ellipsoids to " << path << std::endl;
    }

    void System::SaveTrajectoryTUM(const string &filename)
    {
        cout << endl << "[System] Saving camera trajectory to " << filename << " ..." << endl;

        std::vector<Frame*> vpKFs = mpTracker->GetAllFrames();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for( int i=0;i<vpKFs.size();i++)        
        {
            Frame* pF = vpKFs[i];
            Eigen::Vector3d trans = pF->cam_pose_Twc.translation();
            Eigen::Quaterniond q = pF->cam_pose_Twc.rotation();
            f << setprecision(6) << pF->timestamp << " " <<  setprecision(9) << trans[0] << " " << trans[1] << " " << trans[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        f.close();
        cout << endl << "trajectory saved!" << endl;
    }

    void System::OpenDepthEllipsoid()
    {
        mpTracker->OpenDepthEllipsoid();
    }

    void System::OpenOptimization()
    {
        mpTracker->OpenOptimization();
    }

    void System::CloseOptimization()
    {
        mpTracker->CloseOptimization();
    }

    void System::SetTracker(Tracking* pTracker)
    {
        if(pTracker!=NULL){
            mpTracker = pTracker;
            mpFrameDrawer->setTracker(mpTracker);
        }
        else
        {
            std::cerr << "NULL Tracker, Please check." << std::endl;
        }
    }
}