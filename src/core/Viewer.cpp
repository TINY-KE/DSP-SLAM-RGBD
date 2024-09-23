#include "include/core/Viewer.h"
#include <pangolin/pangolin.h>
#include "src/config/Config.h"
#include "utils/dataprocess_utils.h"

#include <opencv2/core/eigen.hpp>
#include <strstream>  
#include <ostream>  
#include <iostream>  

#include <mutex>
#include <math.h>

pangolin::OpenGlRenderState* pOGLState;

namespace EllipsoidSLAM {

    Viewer::Viewer(System *pSystem, const string &strSettingPath, EllipsoidSLAM::MapDrawer *pMapDrawer){
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];

        mbFinishRequested=false;
        mpSystem = pSystem;
        mpMapDrawer = pMapDrawer;

        miRows = fSettings["Camera.height"];
        miCols = fSettings["Camera.width"];

        mbInverseCameraY = int(fSettings["Viewer.InverseCameraY"]) > 0;

        mvMenuStruct.clear();
        
        mbParamInit = false;
    }

    void eigenMatToOpenGLMat(const Eigen::Matrix4d& matEigen, pangolin::OpenGlMatrix &M)
    {
        cv::Mat matIn;
        cv::eigen2cv(matEigen, matIn);
        if(matEigen.cols()==4)
        {
            cv::Mat Rwc(3,3,CV_64F);
            cv::Mat twc(3,1,CV_64F);
            {
                Rwc = matIn.rowRange(0,3).colRange(0,3);
                twc = matIn.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<double>(0,0);
            M.m[1] = Rwc.at<double>(1,0);
            M.m[2] = Rwc.at<double>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<double>(0,1);
            M.m[5] = Rwc.at<double>(1,1);
            M.m[6] = Rwc.at<double>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<double>(0,2);
            M.m[9] = Rwc.at<double>(1,2);
            M.m[10] = Rwc.at<double>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<double>(0);
            M.m[13] = twc.at<double>(1);
            M.m[14] = twc.at<double>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }

    void func_save_view()
    {
        if(!pOGLState) return;
        std::cout << "Save view!" << std::endl;

        auto mat = pOGLState->GetModelViewMatrix();
        std::cout << mat << std::endl;
        
        ofstream out("./pangolin_view.txt");
        for(int i=0;i<16;i++){
            out << mat.m[i];
            
            if((i+1)%4==0)
                out << std::endl;
            else
                out << " " ;
        }
        out.close();
    }

    void func_load_view()
    {
        if(!pOGLState) return;
        std::cout << "Load view!" << std::endl;

        // pOGLState->Follow(Twc);
        auto eigen_mat = readDataFromFile("./pangolin_view.txt");
        pangolin::OpenGlMatrix gl_mat;
        eigenMatToOpenGLMat(eigen_mat.transpose(), gl_mat);

        std::cout << "gl mat : " << gl_mat << std::endl;

        pOGLState->SetModelViewMatrix(gl_mat);
    }

    Viewer::Viewer(const string &strSettingPath, MapDrawer* pMapDrawer):mpMapDrawer(pMapDrawer) {

        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];

        mbFinishRequested=false;

        miRows = fSettings["Camera.height"];
        miCols = fSettings["Camera.width"];

        mvMenuStruct.clear();

        mbParamInit = false;
    }

    void Viewer::SetFrameDrawer(FrameDrawer* pFrameDrawer)
    {
        mpFrameDrawer = pFrameDrawer;
    }

    void Viewer::run() {
        mbFinished = false;

        pangolin::CreateWindowAndBind("EllipsoidSLAM: Map Viewer", 1024, 768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(220));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);    // 倒数第二个为默认值
        pangolin::Var<bool> menuShowCurrentFrame("menu.Show CurrentFrame", true, true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
        pangolin::Var<bool> menuShowKeyFramesDetail("menu.- Detail", false, true);
        pangolin::Var<bool> menuShowEllipsoids("menu.Show Ellipsoids", true, true);
        pangolin::Var<bool> menuShowEllipsoidsObservation("menu.Ellipsoids-Ob", true, true);
        pangolin::Var<bool> menuShowCuboids("menu. - Show Cuboids", false, true);
        pangolin::Var<bool> menuShowEllipsoidsDetails("menu. - Show Details", true, true);
        pangolin::Var<bool> menuShowPlanes("menu.Show Planes", true, true);
        pangolin::Var<bool> menuShowBoundingboxes("menu.Show Bboxes", false, true);
        pangolin::Var<bool> menuShowConstrainPlanesBbox("menu.ConstrainPlanes-bbox", false, true);
        pangolin::Var<bool> menuShowConstrainPlanesCuboids("menu.ConstrainPlanes-cuboids", false, true);
        pangolin::Var<double> SliderEllipsoidProbThresh("menu.Ellipsoid Prob", 0.3, 0.0, 1.0);
        pangolin::Var<bool> menuShowWorldAxis("menu.Draw World Axis", false, true);

        pangolin::Var<bool> menuAMeaningLessBar("menu.----------", false, false);

        pangolin::Var<bool> menuShowOptimizedTraj("menu.Optimized Traj", true, true);
        pangolin::Var<bool> menuShowGtTraj("menu.Gt Traj", true, true);
        pangolin::Var<bool> menuShowRelationArrow("menu.Relation Arrow", false, true);
        pangolin::Var<int> SliderOdometryWeight("menu.Odometry Weight", 40, 0, 40);
        pangolin::Var<double> SliderPlaneWeight("menu.Plane Weight", 1, 0, 100);
        pangolin::Var<double> SliderPlaneDisSigma("menu.PlaneDisSigma", Config::Get<double>("DataAssociation.PlaneError.DisSigma"), 0.01, 0.5);
        pangolin::Var<int> Slider3DEllipsoidScale("menu.3DEllipsoidScale(10^)", log10(Config::Get<double>("Optimizer.Edges.3DEllipsoid.Scale")), -5, 10);
        pangolin::Var<int> Slider2DEllipsoidScale("menu.2DEllipsoidScale(10^)", log10(Config::Get<double>("Optimizer.Edges.2D.Scale")), -5, 10);
        pangolin::Var<int> SliderUseProbThresh("menu.Use Prob Thresh", 0, 0, 1);
        pangolin::Var<int> SliderOptimizeRelationPlane("menu.OptimizeRelationPlane", 0, 0, 1);
        pangolin::Var<bool> menuShowOptimizedSupPlanes("menu.Optimized SupPlanes", false, true);
        pangolin::Var<bool> menuShowSupPlanesObservation("menu.SupPlanes Observations", false, true);
        pangolin::Var<bool> menuOpenQuadricSLAM("menu.Open QuadricSLAM", false, true);

        pangolin::Var<std::function<void(void)>> menuSaveView("menu.Save View", func_save_view);//设置一个按钮，用于调用function函数
        pangolin::Var<std::function<void(void)>> menuLoadView("menu.Load View", func_load_view);//设置一个按钮，用于调用function函数

        pangolin::Var<bool> menuOpenOptimization("menu.Open Optimization", false, true);

        float up_Y = mbInverseCameraY ? 1.0 : -1.0;

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, -mViewpointY*up_Y, mViewpointZ, 0, 0, 0, 0.0, up_Y, 0.0)
        );
        pOGLState = &s_cam;

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View &d_cam = pangolin::Display("cam")
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -float(miCols) / float(miRows))
                .SetHandler(new pangolin::Handler3D(s_cam));
        
        // Add view for images
        pangolin::View& rgb_image = pangolin::Display("rgb")
        .SetBounds(0,0.3,0.2,0.5,float(miCols) / float(miRows))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);

        pangolin::View& depth_image = pangolin::Display("depth")
        .SetBounds(0,0.3,0.5,0.8,float(miCols) / float(miRows))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        bool bFollow = true;

        pangolin::GlTexture imageTexture(miCols,miRows,GL_RGB,false,0,GL_BGR,GL_UNSIGNED_BYTE);

        while (1) {
            RefreshMenu();  // Deal with dynamic menu bars

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc); // get current camera pose
            
            if(menuFollowCamera && bFollow) // Follow camera
            {
                s_cam.Follow(Twc);
            }
            else if(menuFollowCamera && !bFollow)
            {
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,-mViewpointY*up_Y,mViewpointZ, 0,0,0,0.0,up_Y, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if(!menuFollowCamera && bFollow)
            {
                bFollow = false;
            }

            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);

            // pangolin::glDrawAxis(3);    // draw world coordintates
            if(menuShowCurrentFrame)
                mpMapDrawer->drawCameraState();
                
            if(menuShowKeyFrames)
                mpMapDrawer->drawTrajectory();
            
            if(menuShowKeyFramesDetail)
                mpMapDrawer->drawTrajectoryDetail();    // draw with pose
            

            double ellipsoidProbThresh = SliderEllipsoidProbThresh;

            // draw external cubes of ellipsoids 
            if(menuShowCuboids)
                mpMapDrawer->drawObjects(ellipsoidProbThresh);

            // draw ellipsoids
            if(menuShowEllipsoids)
                mpMapDrawer->drawEllipsoids(ellipsoidProbThresh);

            // draw the result of the single-frame ellipsoid extraction
            if(menuShowEllipsoidsObservation)
                mpMapDrawer->drawObservationEllipsoids(ellipsoidProbThresh);

            // draw planes, including grounplanes and symmetry planes
            if(menuShowPlanes)
                mpMapDrawer->drawPlanes(0); // 0:default.
            if(menuShowOptimizedSupPlanes)
                mpMapDrawer->drawPlanes(1);
            if(menuShowSupPlanesObservation)
                mpMapDrawer->drawPlanes(2);

            if(menuShowBoundingboxes)
                mpMapDrawer->drawBoundingboxes();

            if(menuShowConstrainPlanesBbox)
                mpMapDrawer->drawConstrainPlanes(ellipsoidProbThresh, 0);

            if(menuShowConstrainPlanesCuboids)
                mpMapDrawer->drawConstrainPlanes(ellipsoidProbThresh, 1);

            if(menuShowOptimizedTraj)
                mpMapDrawer->drawTrajectoryWithName("OptimizedTrajectory");

            if(menuShowGtTraj)
                mpMapDrawer->drawTrajectoryWithName("AlignedGroundtruth");      // aligned gt trajectory

            if(menuShowRelationArrow)
                mpMapDrawer->drawArrows();

            mpMapDrawer->drawPoints();  // draw point clouds

            if(menuShowWorldAxis)
                mpMapDrawer->drawAxisNormal();

            if(menuShowEllipsoidsDetails)
                mpMapDrawer->drawEllipsoidsLabelText(ellipsoidProbThresh, menuShowEllipsoids, menuShowEllipsoidsObservation);

            // draw pointclouds with names
            RefreshPointCloudOptions();
            mpMapDrawer->drawPointCloudWithOptions(mmPointCloudOptionMap);
            // mpMapDrawer->drawPointCloudLists();

            // draw images : rgb
            cv::Mat rgb = mpFrameDrawer->getCurrentFrameImage();
            if(!rgb.empty())
            {
                imageTexture.Upload(rgb.data,GL_BGR,GL_UNSIGNED_BYTE);
                //display the image
                rgb_image.Activate();
                glColor3f(1.0,1.0,1.0);
                imageTexture.RenderToViewportFlipY();
            }

            // draw images : depth
            cv::Mat depth = mpFrameDrawer->getCurrentDepthFrameImage();
            if(!depth.empty())
            {
                imageTexture.Upload(depth.data,GL_BGR,GL_UNSIGNED_BYTE);
                //display the image
                depth_image.Activate();
                glColor3f(1.0,1.0,1.0);
                imageTexture.RenderToViewportFlipY();
            }

            // *********** DYNAMIC PARAM CONFIGURATION ****************
            Config::SetValue<double>("DEBUG.ODOM.WEIGHT", SliderOdometryWeight);
            Config::SetValue<double>("DEBUG.PLANE.WEIGHT", SliderPlaneWeight);
            Config::SetValue<double>("Dynamic.Optimizer.UseProbThresh", SliderUseProbThresh);
            Config::SetValue<double>("Dynamic.Optimizer.EllipsoidProbThresh", SliderEllipsoidProbThresh);
            Config::SetValue<double>("DataAssociation.PlaneError.DisSigma", SliderPlaneDisSigma);
            Config::SetValue<double>("DEBUG.MODE.QUADRICSLAM", double(menuOpenQuadricSLAM));
            Config::SetValue<double>("DEBUG.MODE.OPTIMIZATION", double(menuOpenOptimization));

            double real_3DEllipsoidScale = exp10(Slider3DEllipsoidScale);
            Config::SetValue<double>("Optimizer.Edges.3DEllipsoid.Scale", real_3DEllipsoidScale);
            
            double real_2DEllipsoidScale = exp10(Slider2DEllipsoidScale);
            Config::SetValue<double>("Optimizer.Edges.2D.Scale", real_2DEllipsoidScale);
            
            Config::SetValue<double>("Dynamic.Optimizer.OptimizeRelationPlane", double(SliderOptimizeRelationPlane));

            mbParamInit = true; //  参数初始化完毕
            
            pangolin::FinishFrame();

            if (CheckFinish())
                break;
        }

        SetFinish();
    }


    bool Viewer::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Viewer::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    void Viewer::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }


    bool Viewer::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    int Viewer::addDoubleMenu(string name, double min, double max, double def){
        unique_lock<mutex> lock(mMutexFinish);

        MenuStruct menu;
        menu.min = min;
        menu.max = max;
        menu.def = def;
        menu.name = name;
        mvMenuStruct.push_back(menu);

        return mvMenuStruct.size()-1;
    }

    bool Viewer::getValueDoubleMenu(int id, double &value){
        unique_lock<mutex> lock(mMutexFinish);
        if( 0 <= id && 0< mvDoubleMenus.size())
        {
            value = mvDoubleMenus[id]->Get();
            return true;
        }
        else
        {
            return false;
        }
        
    }

    void Viewer::RefreshPointCloudOptions()
    {
        // generate options from mmPointCloudOptionMenus, pointclouds with names will only be drawn when their options are activated.
        std::map<std::string,bool> options;
        for( auto pair : mmPointCloudOptionMenus)
            options.insert(make_pair(pair.first, pair.second->Get()));
        
        mmPointCloudOptionMap.clear();
        mmPointCloudOptionMap = options;
    }

    void Viewer::RefreshMenu(){
        unique_lock<mutex> lock(mMutexFinish);

        // Generate menu bar for every pointcloud in pointcloud list.
        auto pointLists = mpSystem->getMap()->GetPointCloudList();

        // Iterate over the menu and delete the menu if the corresponding clouds are no longer available
        for( auto menuPair = mmPointCloudOptionMenus.begin(); menuPair!=mmPointCloudOptionMenus.end();)
        {
            if(pointLists.find(menuPair->first) == pointLists.end())
            {
                if( menuPair->second !=NULL ){
                    delete menuPair->second;        // destroy the dynamic menu 
                    menuPair->second = NULL;
                }
                menuPair = mmPointCloudOptionMenus.erase(menuPair);  
                continue;
            }
            menuPair++;
        }

        // Iterate over the cloud lists to add new menu.
        for( auto cloudPair: pointLists )
        {
            if(mmPointCloudOptionMenus.find(cloudPair.first) == mmPointCloudOptionMenus.end())
            {
                pangolin::Var<bool>* pMenu = new pangolin::Var<bool>(string("menu.") + cloudPair.first, false, true);
                mmPointCloudOptionMenus.insert(make_pair(cloudPair.first, pMenu));            
            }
        }

        // refresh double bars
        int doubleBarNum = mvDoubleMenus.size();
        int structNum = mvMenuStruct.size();
        if( structNum > 0 && structNum > doubleBarNum )
        {
            for(int i = doubleBarNum; i < structNum; i++)
            {
                pangolin::Var<double>* pMenu = new pangolin::Var<double>(string("menu.")+mvMenuStruct[i].name, mvMenuStruct[i].def, mvMenuStruct[i].min, mvMenuStruct[i].max);
                mvDoubleMenus.push_back(pMenu);
            }
        }

    }

    bool Viewer::isParamInit()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbParamInit;
    }
}

