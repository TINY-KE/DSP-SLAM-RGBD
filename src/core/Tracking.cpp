#include "include/core/Tracking.h"
#include "src/config/Config.h"
#include "utils/dataprocess_utils.h"

#include "include/core/PriorInfer.h"

#include "include/core/SemanticLabel.h"

Eigen::MatrixXd matSymPlanes;
PriFactor mPrifac;

namespace EllipsoidSLAM
{
    void outputObjectObservations(std::map<int, Observations> &mmObjectObservations)
    {
        ofstream out("./log_mmObjectObservations.txt");
        
        out << " --------- ObjectObservations : " << std::endl;
        for ( auto obPair: mmObjectObservations)
        {
            out << " ---- Instance " << obPair.first << " (" << obPair.second.size() << ") :" << std::endl;

            for( auto ob : obPair.second )
            {
                out << " -- ob : " << ob->pFrame->frame_seq_id << " | " << ob->bbox.transpose() << " | " << ob->label << " | " << ob->rate << std::endl;
            }

            out << std::endl;
        }

        out.close();
        std::cout << "Save to log_mmObjectObservations.txt..." << std::endl;
    }

    void Tracking::outputBboxMatWithAssociation()
    {
        std::map<double, Observations> mapTimestampToObservations;

        for ( auto obPair: mmObjectObservations)
        {
            int instance = obPair.first;
            for( auto ob : obPair.second )
            {
                ob->instance = instance;

                // save with timestamp
                if(mapTimestampToObservations.find(ob->pFrame->timestamp)!=mapTimestampToObservations.end())
                    mapTimestampToObservations[ob->pFrame->timestamp].push_back(ob);
                else {
                    mapTimestampToObservations.insert(make_pair(ob->pFrame->timestamp, Observations()));
                    mapTimestampToObservations[ob->pFrame->timestamp].push_back(ob);
                }
            }
        }

        for( auto frameObsPair : mapTimestampToObservations ){
            string str_timestamp = to_string(frameObsPair.first);

            string filename = string("./bbox/") + str_timestamp + ".txt";
            ofstream out(filename.c_str());
            
            int num = 0;
            for ( auto ob: frameObsPair.second)
            {
                
                out << num++ << " " << ob->bbox.transpose() << " " << ob->label << " " << ob->rate << " " << ob->instance << std::endl;
            }

            out.close();
            std::cout << "Save to " << filename << std::endl;
        }

        std::cout << "Finish... " << std::endl;
                
    }

    Tracking::Tracking(EllipsoidSLAM::System *pSys, EllipsoidSLAM::FrameDrawer *pFrameDrawer,
                       EllipsoidSLAM::MapDrawer *pMapDrawer, EllipsoidSLAM::Map *pMap, const string &strSettingPath)
                       :mpMap(pMap), mpSystem(pSys), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer)
   {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4,1,CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if(k3!=0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if(fps==0)
            fps=30;

        int rows = fSettings["Camera.height"];
        int cols = fSettings["Camera.width"];

        mCalib << fx,  0,  cx,
                0,  fy, cy,
                0,      0,     1;

        mCamera.cx = cx;
        mCamera.cy = cy;
        mCamera.fx = fx;
        mCamera.fy = fy;
        mCamera.scale = fSettings["Camera.scale"];

        mpInitializer =  new Initializer(rows, cols);
        mpOptimizer = new Optimizer;
        mRows = rows;
        mCols = cols;

        mbDepthEllipsoidOpened = false;

        mbOpenOptimization = true;

        pDASolver = new DataAssociationSolver(mpMap);
        mpBuilder = new Builder();
        mpBuilder->setCameraIntrinsic(mCalib, mCamera.scale);

        mCurrFrame = NULL;

        // output
        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if(DistCoef.rows==5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;
        cout << "- rows: " << rows << endl;
        cout << "- cols: " << cols << endl;
        cout << "- Scale: " << mCamera.scale << endl;

        // ********** DEBUG ***********
        matSymPlanes.resize(0, 5);
        mpRelationExtractor = new RelationExtractor();

        // 创建一个 PriFactor
        const string priconfig_path = Config::Get<std::string>("Dataset.Path.PriTable");
        bool use_input_pri = (priconfig_path.size()>0);
        if(use_input_pri){
            mPrifac.LoadPriConfigurations(priconfig_path);
        }
    }

    g2o::ellipsoid* Tracking::getObjectDataAssociation(const Eigen::VectorXd &pose, const Eigen::VectorXd &detection) {
        auto objects = mpMap->GetAllEllipsoids();
        if( objects.size() > 0 )
            return objects[0];
        else
            return NULL;
    }

    bool Tracking::GrabPoseAndObjects(const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
    const cv::Mat &imDepth, const cv::Mat &imRGB, bool withAssociation) {
        return GrabPoseAndObjects(0, pose, bboxMap, imDepth, imRGB, withAssociation);
    }

    cv::Mat FilterDepthMat(const cv::Mat &imDepth)
    {
        cv::Mat imDepthFiltered;
        int filter_param = Config::Get<int>("Median.Filter.Param");
        std::cout << "MEDIAN_FILTER param: " << filter_param << std::endl;
        medianBlur(imDepth, imDepthFiltered, filter_param);

        cv::Mat imDepthFiltered_32F;
        imDepthFiltered.convertTo(imDepthFiltered_32F, CV_32F);

        // Gaussian
        cv::Mat imDepthFiltered2;
        int filter_bilateral_param = Config::Get<int>("Median.Filter.Bilateral");
        std::cout << "Median.Filter.Bilateral param: " << filter_bilateral_param << std::endl;

        bilateralFilter( imDepthFiltered_32F, imDepthFiltered2, filter_bilateral_param, filter_bilateral_param*2, filter_bilateral_param/2 );

        cv::Mat imDepth2_16U;
        imDepthFiltered2.convertTo(imDepth2_16U, CV_16U);


        return imDepth2_16U;
    }

    void Tracking::AddNewFrame(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
        const cv::Mat &imDepth, const cv::Mat &imRGB)
    {
        // 数据预处理

        // [ 功能 : 中值滤波去噪 ]
        // 暂时关闭. 尽量避免引入过多参数.
        // 若启用, 注意函数要切换到此.
        // cv::Mat imDepthFiltered = FilterDepthMat(imDepth);

        Frame *pF = new Frame(timestamp, pose, bboxMap, imDepth, imRGB, true);
        mvpFrames.push_back(pF);
        mCurrFrame = pF;
    }

    bool Tracking::DealWithOnline()
    {
        double thresh_time_diff_s = 0.5; // 2Hz!

        // --------------
        bool bKeyFrame = true;

        static double last_key_timestamp = 0;
        if(mvpFrames.size() > 1)
        {
            Frame* pLastFrame = mvpFrames.back(); 
            double current_timestamp = pLastFrame->timestamp;
            if( (current_timestamp - last_key_timestamp) < thresh_time_diff_s )
                bKeyFrame = false;
            else 
                last_key_timestamp = current_timestamp;
        }

        // ------------
        // 处理后端优化的交互开关

        // 第一次启动触发清空 KeyFrames
        static bool bFirstInit = true;
        // mbOpenOptimization : 静态
        // 动态开关
        bool mbDynamicOpenOptimization = Config::ReadValue<double>("DEBUG.MODE.OPTIMIZATION") > 0.5;
        if(!mbDynamicOpenOptimization) 
        {
            bKeyFrame = true;
            bFirstInit = true;
        }
        
        if(mbDynamicOpenOptimization && bFirstInit)
        {
            mvpFrames.clear();
            Frame::Clear();
            bFirstInit = false;
        }

        return bKeyFrame;
    }

    bool Tracking::GrabPoseAndObjects(double timestamp, const Eigen::VectorXd &pose, const Eigen::MatrixXd &bboxMap,
        const cv::Mat &imDepth, const cv::Mat &imRGB, bool withAssociation) {
        clock_t time_start = clock();        
        
        // ZHJD:将新帧的时间戳、位姿、边界框（物体位置）、深度图和 RGB 图像添加到系统中进行处理。
        AddNewFrame(timestamp, pose, bboxMap, imDepth, imRGB);
        clock_t time_AddNewFrame = clock();

        bool bOnlineRunning = Config::Get<int>("System.Online.Open") > 0; // 现场实时运行所用，取指定时间间隔作为关键帧; 跑数据集请保持关闭以处理所有帧
        bool mbDynamicOpenOptimization = Config::ReadValue<double>("DEBUG.MODE.OPTIMIZATION") > 0.5;
        bool bKeyFrame;      
        if(bOnlineRunning)
            bKeyFrame = DealWithOnline();
        else 
            bKeyFrame = true;   // 否则处理所有帧


        // ZHJD: 如果当前帧是关键帧，则更新对象观察，存储对象观察到的数据。
        clock_t time_UpdateObjectObservation, time_NonparamOptimization;
        if(bKeyFrame){
            UpdateObjectObservation(mCurrFrame, withAssociation);   // Store object observation in a specific data structure.

            time_UpdateObjectObservation = clock();

            // ZHJD 如果动态优化开启，则进行非参数优化，优化数据关联、对象和地标。
            if(mbDynamicOpenOptimization){
                NonparamOptimization(); // Optimize data associations,objects,landmarks.        
            }
            time_NonparamOptimization = clock();
        }
        else 
            std::cout << "Not Key Frame." << std::endl;

        // Visualization
        ProcessVisualization();
        clock_t time_Visualization = clock();

        // Memory Management
        // 释放掉两帧前的 cv::Mat , rgb/depth.
        ManageMemory();

        // Output running time
        if(bKeyFrame)
        {
            cout << " - System Time: " << endl;
            cout << " -- time_AddNewFrame: " <<(double)(time_AddNewFrame - time_start) / CLOCKS_PER_SEC << "s" << endl;        
            cout << " -- time_UpdateObjectObservation: " <<(double)(time_UpdateObjectObservation - time_AddNewFrame) / CLOCKS_PER_SEC << "s" << endl;
            cout << " -- time_NonparamOptimization: " <<(double)(time_NonparamOptimization - time_UpdateObjectObservation) / CLOCKS_PER_SEC << "s" << endl;
            cout << " -- time_Visualization: " <<(double)(time_Visualization - time_NonparamOptimization) / CLOCKS_PER_SEC << "s" << endl;
            cout << " - [ total_frame: " <<(double)(time_Visualization - time_start) / CLOCKS_PER_SEC << "s ]" << endl;
        }
        return true;
    }

    void Tracking::ProcessVisualization()
    {
        // Visualize frames with intervals
        if( isKeyFrameForVisualization() )
            mpMap->addCameraStateToTrajectory(&mCurrFrame->cam_pose_Twc);
        mpMap->setCameraState(&mCurrFrame->cam_pose_Twc);

        // Render rgb images and depth images for visualization.
        cv::Mat imForShow = mpFrameDrawer->drawFrame();
        // cv::Mat imForShowDepth = mpFrameDrawer->drawDepthFrame();
        cv::Mat imForShowDepthWithVisualPoints = mpFrameDrawer->drawDepthFrameWithVisualPoints();   // 包含点云分割过程的可视化的图.
        
        // [A visualization tool] When Builder is opened, it generates local pointcloud from depth and rgb images of current frame,
        // and global pointcloud by simply adding local pointcloud in world coordinate and then downsampling them for visualization. 
        bool mbOpenBuilder = Config::Get<int>("Visualization.Builder.Open") > 0;
        if(mbOpenBuilder)
        {
            double depth_range = Config::ReadValue<double>("EllipsoidExtractor_DEPTH_RANGE");   // Only consider pointcloud within depth_range

            if(!mCurrFrame->rgb_img.empty()){    // RGB images are needed.
                Eigen::VectorXd pose = mCurrFrame->cam_pose_Twc.toVector();
                mpBuilder->processFrame(mCurrFrame->rgb_img, mCurrFrame->frame_img, pose, depth_range);

                mpBuilder->voxelFilter(0.01);   // Down sample threshold; smaller the finer; depend on the hardware.
                PointCloudPCL::Ptr pCloudPCL = mpBuilder->getMap();
                PointCloudPCL::Ptr pCurrentCloudPCL = mpBuilder->getCurrentMap();

                auto pCloud = pclToQuadricPointCloudPtr(pCloudPCL);
                auto pCloudLocal = pclToQuadricPointCloudPtr(pCurrentCloudPCL);
                mpMap->AddPointCloudList("Builder.Global Points", pCloud);
                mpMap->AddPointCloudList("Builder.Local Points", pCloudLocal);
            }
        }
    }

    void AddSegCloudsToQuadricStorage(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& segClouds, EllipsoidSLAM::PointCloud* pSegCloud){
        int cloud_num = segClouds.size();
        srand(time(0));
        for(int i=0;i<cloud_num;i++)
        {
            int point_num = segClouds[i]->points.size();
            int r = rand()%155;
            int g = rand()%155;
            int b = rand()%155;
            for( int n=0;n<point_num;n++)
            {
                PointXYZRGB point;
                point.x =  segClouds[i]->points[n].x;
                point.y =  segClouds[i]->points[n].y;
                point.z =  segClouds[i]->points[n].z;
                point.r = r;
                point.g = g;
                point.b = b;

                point.size = 2;
                pSegCloud->push_back(point);
            }
            
        }

        return;

    }

    // Debug函数用于可视化
    void VisualizeCuboidsPlanesInImages(g2o::ellipsoid& e, const g2o::SE3Quat& campose_wc, const Matrix3d& calib, int rows, int cols, Map* pMap)
    {
        g2o::ellipsoid e_global = e.transform_from(campose_wc);
        Vector3d center = e_global.pose.translation();

        std::vector<plane*> pPlanes = e.GetCubePlanesInImages(g2o::SE3Quat(), calib, rows, cols, 30);
        int planeNum = pPlanes.size();
        for( int i=0; i<planeNum; i++){
            Vector4d planeVec = pPlanes[i]->param.head(4);
            Vector3d color(0,0,1.0);  
            double plane_size = e.scale.norm()/2.0;

            g2o::plane *pPlane = new g2o::plane(planeVec, color);
            pPlane->transform(campose_wc);
            pPlane->InitFinitePlane(center, plane_size);
            pMap->addPlane(pPlane);
        }
    }

    void Tracking::ClearVisualization()
    {
        mpMap->ClearEllipsoidsObservation(); // Clear the Visual Ellipsoids in the map
        // mpMap->ClearEllipsoidsVisual(); // Clear the Visual Ellipsoids in the map
        mpMap->ClearBoundingboxes();
        mpMap->clearPlanes();

        if(miGroundPlaneState == 2) // if the groundplane has been estimated
            mpMap->addPlane(&mGroundPlane);

        VisualizeManhattanPlanes();
    }

    // Process Ellipsoid Estimation for every boundingboxes in current frame.
    // Finally, store 3d Ellipsoids into the member variable mpLocalObjects of pFrame.
    // 对当前帧中的每个边界框进行 Process Ellipsoid Estimation。
    // 最后，将 3D 椭球体存储到 pFrame 的成员变量 mpLocalObjects 中。
    void Tracking::UpdateDepthEllipsoidEstimation(EllipsoidSLAM::Frame* pFrame, bool withAssociation)
    {
        if( !mbDepthEllipsoidOpened ) return;

        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int rows = obs_mat.rows();

        Eigen::VectorXd pose = pFrame->cam_pose_Twc.toVector();

        mpEllipsoidExtractor->ClearPointCloudList();    // clear point cloud visualization

        bool bPlaneNotClear = true;
        bool bEllipsoidNotClear = true;
        for(int i=0;i<rows;i++){
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate instanceID
            int label = round(det_vec(5));
            double measurement_prob = det_vec(6);

            Eigen::Vector4d measurement = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            // Filter those detections lying on the border.
            bool is_border = calibrateMeasurement(measurement, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            double prob_thresh = Config::Get<double>("Measurement.Probability.Thresh");
            bool prob_check = (measurement_prob > prob_thresh);

            g2o::ellipsoid* pEllipsoidForThisObservation = NULL;
            // 2 conditions must meet to start ellipsoid extraction:
            // C1 : the bounding box is not on border
            bool c1 = !is_border;

            // C2 : the groundplane has been estimated successfully
            bool c2 = miGroundPlaneState == 2;
            
            // in condition 3, it will not start
            // C3 : under with association mode, and the association is invalid, no need to extract ellipsoids again.
            bool c3 = false;
            if( withAssociation )
            {
                int instance = round(det_vec(7));
                if ( instance < 0 ) c3 = true;  // invalid instance
            }

            // C4 : 物体过滤
            // 部分动态物体，如人类， label=0，将被过滤不考虑
            bool c4 = true;
            std::set<int> viIgnoreLabelLists = {
                0 // Human
            };
            if(viIgnoreLabelLists.find(label) != viIgnoreLabelLists.end())
                c4 = false;

            if( prob_check && c1 && c2 && !c3 && c4 ){   
                if(bPlaneNotClear){
                    mpMap->clearPlanes();
                    if(miGroundPlaneState == 2) // if the groundplane has been estimated
                        mpMap->addPlane(&mGroundPlane);

                    VisualizeManhattanPlanes();
                    bPlaneNotClear = false;
                }
                g2o::ellipsoid e_extractByFitting_newSym = mpEllipsoidExtractor->EstimateLocalEllipsoidUsingMultiPlanes(pFrame->frame_img, measurement, label, measurement_prob, pose, mCamera);
                
                bool c0 = mpEllipsoidExtractor->GetResult();
                // 可视化部分
                if( c0 )
                {
                    // Visualize estimated ellipsoid
                    g2o::ellipsoid* pObjByFitting = new g2o::ellipsoid(e_extractByFitting_newSym.transform_from(pFrame->cam_pose_Twc));
                    if(pObjByFitting->prob_3d > 0.5)
                        pObjByFitting->setColor(Vector3d(0.8,0.0,0.0), 1); // Set green color
                    else 
                        pObjByFitting->setColor(Vector3d(0.8,0,0), 0.5); // 透明颜色

                    // 临时更新： 此处显示的是 3d prob
                    // pObjByFitting->prob = pObjByFitting->prob_3d;

                    // 第一次添加时清除上一次观测!
                    if(bEllipsoidNotClear)
                    {
                        mpMap->ClearEllipsoidsVisual(); // Clear the Visual Ellipsoids in the map
                        mpMap->ClearBoundingboxes();
                        bEllipsoidNotClear = false;
                    }
                    mpMap->addEllipsoidVisual(pObjByFitting);

                    // 添加debug, 测试筛选图像平面内的bbox平面
                    VisualizeCuboidsPlanesInImages(e_extractByFitting_newSym, pFrame->cam_pose_Twc, mCalib, mRows, mCols, mpMap);

                }   // successful estimation.

                // 存储条件1: 该检测 3d_prob > 0.5
                // 最终决定使用的估计结果
                if( c0 ){
                    g2o::ellipsoid *pE_extractByFitting = new g2o::ellipsoid(e_extractByFitting_newSym);
                    pEllipsoidForThisObservation = pE_extractByFitting;   // Store result to pE_extracted.
                }
            }

            // 若不成功保持为NULL
            pFrame->mpLocalObjects.push_back(pEllipsoidForThisObservation);
        }

        return;
    }

    void Tracking::Update3DObservationDataAssociation(EllipsoidSLAM::Frame* pFrame, std::vector<int>& associations, std::vector<bool>& KeyFrameChecks)
    {
        int num = associations.size();

        if( mbDepthEllipsoidOpened )
        {
            std::vector<g2o::ellipsoid*> pLocalObjects = pFrame->mpLocalObjects;

            for( int i=0; i<num; i++)
            {
                if(pLocalObjects[i] == NULL )   // if the single-frame ellipsoid estimation fails
                    continue;

                int instance = associations[i];
                if(instance < 0 ) continue; // if the data association is invalid

                if( !KeyFrameChecks[i] )  // if the observation for the object is not key observation (without enough intervals to the last observation).
                {
                    pFrame->mpLocalObjects[i] = NULL;
                    continue;
                }

                // Save 3D observations
                Observation3D* pOb3d = new Observation3D;
                pOb3d->pFrame = pFrame;
                pOb3d->pObj = pLocalObjects[i];
                mmObjectObservations3D[instance].push_back(pOb3d);

                // Set instance to the ellipsoid according to the associations
                pLocalObjects[i]->miInstanceID = instance;
            }
        }

        return;
    }

    // Consider key observations for every object instances.
    // key observations: two valid observations for the same instance should have enough intervals( distance or angles between the two poses ).
    std::vector<bool> Tracking::checkKeyFrameForInstances(std::vector<int>& associations)
    {
        double CONFIG_KEYFRAME_DIS;
        double CONFIG_KEYFRAME_ANGLE;

        if( Config::Get<int>("Tracking.KeyFrameCheck.Close") == 1)
        {
            CONFIG_KEYFRAME_DIS = 0;  
            CONFIG_KEYFRAME_ANGLE = 0; 
        }
        else
        {
            CONFIG_KEYFRAME_DIS = 0.4;  
            CONFIG_KEYFRAME_ANGLE = CV_PI/180.0*15; 
        }

        int num =associations.size();
        std::vector<bool> checks; checks.resize(num);
        fill(checks.begin(), checks.end(), false);
        for( int i=0;i<num;i++)
        {
            int instance = associations[i];
            if(instance<0) 
            {
                checks[i] = false;
            }
            else
            {
                if(mmObjectObservations.find(instance) == mmObjectObservations.end())   // if the instance has not been initialized
                {
                    checks[i] = true;
                }
                else
                {
                    Observations &obs = mmObjectObservations[instance];
                    // Get last frame
                    g2o::SE3Quat &pose_last_wc = obs.back()->pFrame->cam_pose_Twc;
                    g2o::SE3Quat &pose_curr_wc = mCurrFrame->cam_pose_Twc;

                    g2o::SE3Quat pose_diff = pose_curr_wc.inverse() * pose_last_wc;
                    double dis = pose_diff.translation().norm();

                    Eigen::Quaterniond quat = pose_diff.rotation();
                    Eigen::AngleAxisd axis(quat);
                    double angle = axis.angle();

                    if( dis > CONFIG_KEYFRAME_DIS || angle > CONFIG_KEYFRAME_ANGLE)
                        checks[i] = true;
                    else
                        checks[i] = false;
                }
            }
        }
        return checks;
    }

    // for the mannual data association, 
    // this function will directly return the results of [instance] in the object detection matrix
    // PS. one row of detection matrix is : id x1 y1 x2 y2 label rate instanceID
    std::vector<int> Tracking::GetMannualAssociation(Eigen::MatrixXd &obsMat)
    {
        int num = obsMat.rows();
        std::vector<int> associations; associations.resize(num);
        for( int i=0; i<num; i++)
        {
            VectorXd vec = obsMat.row(i);
            associations[i] = round(vec[7]);
        }
        
        return associations;
    }

    void Tracking::ActivateGroundPlane(g2o::plane &groundplane)
    {
        // Set groundplane to EllipsoidExtractor
        if( mbDepthEllipsoidOpened ){
            std::cout << " * Add supporting plane to Ellipsoid Extractor." << std::endl;
            mpEllipsoidExtractor->SetSupportingPlane(&groundplane, false);
        }

        // Set groundplane to Optimizer
        std::cout << " * Add supporting plane to optimizer. " << std::endl;
        mpOptimizer->SetGroundPlane(groundplane.param);

        std::cout << " * Add supporting plane to Manhattan Plane Extractor. " << std::endl;
        pPlaneExtractorManhattan->SetGroundPlane(&groundplane);

        // Change state
        miGroundPlaneState = 2;
    }

    void Tracking::SetGroundPlaneMannually(const Eigen::Vector4d &param)
    {
        std::cout << "[GroundPlane] Set groundplane mannually: " << param.transpose() << std::endl;
        miGroundPlaneState = 3;
        mGroundPlane.param = param;
        mGroundPlane.color = Vector3d(0,1,0);
    }

    void Tracking::TaskGroundPlane()
    {
        // int miGroundPlaneState; // 0: Closed  1: estimating 2: estimated 3: set by mannual
        if(miGroundPlaneState == 1) // State 1: Groundplane estimation opened, and not done yet.
            ProcessGroundPlaneEstimation();
        else if(miGroundPlaneState == 3) // State : Set by mannual
            ActivateGroundPlane(mGroundPlane);

    }

    void Tracking::VisualizeManhattanPlanes()
    {
        // 可视化提取结果.
        std::vector<g2o::plane*> vDominantMHPlanes = pPlaneExtractorManhattan->GetDominantMHPlanes();
        for(auto& vP:vDominantMHPlanes ) {
            mpMap->addPlane(vP);
        }
    }

    // running after groundplane has been estimated
    void Tracking::TaskManhattanPlanes(EllipsoidSLAM::Frame *pFrame)
    {
        // 清除可视化
        mpMap->DeletePointCloudList("pPlaneExtractorManhattan", 1);
        g2o::SE3Quat Twc = pFrame->cam_pose_Twc;

        // int miGroundPlaneState; // 0: Closed  1: estimating 2: estimated 3: set by mannual
        if(miGroundPlaneState == 2 && miMHPlanesState == 1){
            g2o::plane local_ground = mGroundPlane;
            local_ground.transform(pFrame->cam_pose_Tcw);
            Vector3d local_gt = local_ground.param.head(3);
            bool result = pPlaneExtractorManhattan->extractManhattanPlanes(pFrame->frame_img, local_gt, Twc);
            
            if(mbDepthEllipsoidOpened && result)    // 仅当开启了 Depth 提取.
            {
                // 动态更新 EllipsoidExtractor 中存放的平面
                // 仅当新增了新的 MH Planes时，该返回值为真.
                if( pPlaneExtractorManhattan->GetMHResult())
                {
                    // 在此添加一个 MHPlanes 的开关. 关闭后，MHPlane的分割不再生效，但支撑平面会继续生效. 
                    bool bOpenMHPlane = Config::Get<int>("Plane.ManhattanPlane.Open") > 0;
                    if(!bOpenMHPlane) return;

                    mpEllipsoidExtractor->SetManhattanPlanes(pPlaneExtractorManhattan->GetDominantMHPlanes());
                }
            }

            // 可视化
            bool bVisualizePlanes = true;
            if(bVisualizePlanes)
            {
                std::vector<PointCloudPCL> vpPlanePoints = pPlaneExtractorManhattan->GetPoints();
                mpMap->AddPointCloudList("pPlaneExtractorManhattan.MH-Plane Points", vpPlanePoints, Twc, REPLACE_POINT_CLOUD);
            }

            bool bVisualizeMHPlanes = true;
            if(bVisualizeMHPlanes)
            {
                std::vector<PointCloudPCL> vpPlanePoints = pPlaneExtractorManhattan->GetPotentialGroundPlanePoints();
                mpMap->AddPointCloudList("pPlaneExtractorManhattan.MH-MH Points", vpPlanePoints, Twc, REPLACE_POINT_CLOUD);
            }
        }

        return;
    }

    void VisualizeRelations(Relations& rls, Map* pMap, g2o::SE3Quat &Twc, std::vector<PointCloudPCL>& vPlanePoints)
    {
        int num = rls.size();
        // std::cout << "Relation Num: " << num << std::endl;
        
        int mode = 0;   //clear

        pMap->clearArrows();
        for(int i=0;i<num;i++)
        {
            Relation &rl = rls[i];
            g2o::ellipsoid* pEllip = rl.pEllipsoid;
            g2o::plane* pPlane = rl.pPlane;
            if(pEllip==NULL || pPlane==NULL) {
                std::cout << "[Relation] NULL relation : " << rl.obj_id << ", " << rl.plane_id << std::endl;
                continue;
            }
            g2o::ellipsoid e_world = pEllip->transform_from(Twc);
            g2o::plane* plane_world = new g2o::plane(*pPlane); plane_world->transform(Twc);
            Vector3d obj_center = e_world.pose.translation();
            Vector3d norm = plane_world->param.head(3); norm.normalize();
            double length = 0.5; norm = norm * length;

            if(rl.type == 1)    // 支撑
            {
                // 即在物体底端产生一个向上大竖直箭头.
                // 以物体为中心.
                // 以平面法向量为方向.
                pMap->addArrow(obj_center, norm, Vector3d(0,1.0,0));
            }
            else if(rl.type == 2) // 倚靠
            {
                // 同上
                pMap->addArrow(obj_center, norm, Vector3d(0,0,1.0));
            }

            // 同时高亮平面.
            plane_world->InitFinitePlane(obj_center, 0.7);
            plane_world->color = Vector3d(0, 1, 1);    // 黄色显示关系面
            pMap->addPlane(plane_world);

            // 高亮对应平面的点云
            int plane_id = rl.plane_id;
            if(plane_id >= 0 && plane_id < vPlanePoints.size())
            {
                PointCloudPCL::Ptr pCloudPCL(new PointCloudPCL(vPlanePoints[rl.plane_id]));
                EllipsoidSLAM::PointCloud cloudQuadri = pclToQuadricPointCloud(pCloudPCL);
                EllipsoidSLAM::PointCloud* pCloudGlobal = transformPointCloud(&cloudQuadri, &Twc);
                
                int r = 0;
                int g = 255;
                int b = 255;
                SetPointCloudProperty(pCloudGlobal, r, g, b, 4);
                pMap->AddPointCloudList(string("Relationship.Activiate Sup-Planes"), pCloudGlobal, mode);
                if(mode == 1){
                    delete pCloudGlobal;    // 该指针对应的点云已被拷贝到另一个指针点云,清除多余的一个
                    pCloudGlobal = NULL;
                }

                mode = 1;   // 仅仅第一次清除.
            }
            else 
            {
                std::cout << "Invalid plane_id : " << plane_id << std::endl;
            }
            
        }
    }

    void VisualizeBottomPlane(std::vector<g2o::ellipsoid*>& vpEllipsoids, Map *mpMap, Frame * pFrame)
    {
        int num = vpEllipsoids.size();

        g2o::SE3Quat Twc = pFrame->cam_pose_Twc;
        for(int i=0;i<num;i++)
        {
            g2o::ellipsoid* pEllip = vpEllipsoids[i];
            if(pEllip==NULL) continue;

            auto e_world = pEllip->transform_from(Twc); // 到世界系
            std::vector<g2o::plane*> vpPlanes = e_world.GetCubePlanes();

            auto pPl = vpPlanes[0]; //可视化id为0的面. 
            pPl->color = Vector3d(0,1.0,1.0);   // yellow
            pPl->InitFinitePlane(e_world.pose.translation(), 1);
            mpMap->addPlane(pPl);
        }
    }

    // *******
    // 1) 基于局部提取的平面，做一次分割以及椭球体提取
    // 2) 若该椭球体满足 IoU >0.5, 则替换掉之前的
    // 3) 若不满足，则使用点云中心+bbox产生点模型椭球体
    void Tracking::RefineObjectsWithRelations(EllipsoidSLAM::Frame *pFrame)
    {
        Relations& rls = pFrame->relations;
        int num = rls.size();

        Eigen::VectorXd pose = pFrame->cam_pose_Twc.toVector();

        int success_num = 0;
        for(int i=0;i<num;i++){
            // 对于支撑关系, 且平面非地平面
            // 将该新平面加入到 MHPlanes 中，重新计算一遍提取.
            Relation& rl = rls[i];
            if(rl.type == 1){   // 支撑关系
                g2o::plane* pSupPlane = rl.pPlane;  // 局部坐标系的平面位置. TODO: 检查符号

                int obj_id = rl.obj_id;

                // 此处需要bbox位置.
                // cv::Mat& depth, Eigen::Vector4d& bbox, int label, double prob, Eigen::VectorXd &pose, camera_intrinsic& camera
                Eigen::VectorXd det_vec = pFrame->mmObservations.row(obj_id);  // id x1 y1 x2 y2 label rate imageID
                int label = round(det_vec(5));
                Eigen::Vector4d bbox = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));
                double prob = det_vec(6);

                g2o::ellipsoid e = mpEllipsoidExtractor->EstimateLocalEllipsoidWithSupportingPlane(pFrame->frame_img, bbox, label, prob, pose, mCamera, pSupPlane); // 取消
                // 该提取不再放入 world? 不, world MHPlanes 还是需要考虑的.

                // 可视化该 Refined Object
                bool c0 = mpEllipsoidExtractor->GetResult();
                if( c0 )
                {
                    // Visualize estimated ellipsoid
                    g2o::ellipsoid* pObjRefined = new g2o::ellipsoid(e.transform_from(pFrame->cam_pose_Twc));
                    pObjRefined->setColor(Vector3d(0,0.8,0), 1); 
                    mpMap->addEllipsoidVisual(pObjRefined);

                    // 存储条件1: 该检测 3d_prob > 0.5
                    // bool c1 = (e.prob_3d > 0.5);
                    // 最终决定使用的估计结果
                    // if( c0 && c1 ){
                        // (*pFrame->mpLocalObjects[obj_id]) = e;                    
                        // success_num++;

                    // }

                    // 此处设定 Refine 一定优先.
                    (*pFrame->mpLocalObjects[obj_id]) = e;                    
                    success_num++;

                }
            }
        }
        std::cout << "Refine result : " << success_num << " objs." << std::endl;
    }


    void Tracking::TaskRelationshipWithPointModel(EllipsoidSLAM::Frame *pFrame)
    {
        std::vector<g2o::ellipsoid*>& vpEllipsoids = pFrame->mpLocalObjects;
        // 获得局部 planes.
        std::vector<g2o::plane*> vpPlanes = pPlaneExtractorManhattan->GetPotentialMHPlanes();

        Relations rls = mpRelationExtractor->ExtractSupporttingRelations(vpEllipsoids, vpPlanes, pFrame, POINT_MODEL);

        if(rls.size()>0)
        {
            // 将结果存储到 frame 中
            pFrame->mbSetRelation = true;
            pFrame->relations = rls;
        }

        // ****************************
        //          可视化部分
        // ****************************
        g2o::SE3Quat Twc = pFrame->cam_pose_Twc;
        std::vector<PointCloudPCL> vPlanePoints = pPlaneExtractorManhattan->GetPotentialMHPlanesPoints();
        mpMap->AddPointCloudList("Relationship.Relation Planes", vPlanePoints, Twc, REPLACE_POINT_CLOUD);

        // 可视化该关系
        VisualizeRelations(rls, mpMap, Twc, vPlanePoints); // 放到地图中去显示?

        // std::cout << "Objects: " << vpEllipsoids.size() << std::endl;
        // std::cout << "Relation Planes : " << vpPlanes.size() << std::endl;
        // std::cout << "Relations : " << rls.size() << std::endl;
    }

    void Tracking::TaskRelationship(EllipsoidSLAM::Frame *pFrame)
    {
        std::vector<g2o::ellipsoid*>& vpEllipsoids = pFrame->mpLocalObjects;
        // 获得局部 planes.
        std::vector<g2o::plane*> vpPlanes = pPlaneExtractorManhattan->GetPotentialMHPlanes();

        Relations rls = mpRelationExtractor->ExtractSupporttingRelations(vpEllipsoids, vpPlanes, pFrame, QUADRIC_MODEL);

        if(rls.size()>0)
        {
            // 将结果存储到 frame 中
            pFrame->mbSetRelation = true;
            pFrame->relations = rls;
        }

        // ****************************
        //          可视化部分
        // ****************************
        g2o::SE3Quat Twc = pFrame->cam_pose_Twc;
        std::vector<PointCloudPCL> vPlanePoints = pPlaneExtractorManhattan->GetPotentialMHPlanesPoints();
        mpMap->AddPointCloudList("Relationship.Relation Planes", vPlanePoints, Twc, REPLACE_POINT_CLOUD);

        // 可视化该关系
        VisualizeRelations(rls, mpMap, Twc, vPlanePoints); // 放到地图中去显示?

        // std::cout << "Objects: " << vpEllipsoids.size() << std::endl;
        // std::cout << "Relation Planes : " << vpPlanes.size() << std::endl;
        // std::cout << "Relations : " << rls.size() << std::endl;
    }

    void Tracking::ManageMemory()
    {
        if(mvpFrames.size() > 1){
            Frame* pLastFrame = mvpFrames[mvpFrames.size()-2];

            // std::cout << "Going to release image..." << std::endl;
            // getchar();
            pLastFrame->frame_img.release();
            pLastFrame->rgb_img.release();
            // std::cout << "Released rgb and depth images." << std::endl;
        }

    }


    void Tracking::UpdateObjectObservationWithPointModel(Frame* pFrame, bool withAssociation)
    {
        if( !mbDepthEllipsoidOpened ) {
            std::cout << "Ellipsoid estimation closed." << std::endl;
            return;
        }

        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int rows = obs_mat.rows();

        Eigen::VectorXd pose = pFrame->cam_pose_Twc.toVector();

        bool bPlaneNotClear = true;
        bool bEllipsoidNotClear = true;
        for(int i=0;i<rows;i++){
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate instanceID
            int label = round(det_vec(5));
            double measurement_prob = det_vec(6);

            Eigen::Vector4d measurement = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            // Filter those detections lying on the border.
            bool is_border = calibrateMeasurement(measurement, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            double prob_thresh = Config::Get<double>("Measurement.Probability.Thresh");
            bool prob_check = (measurement_prob > prob_thresh);

            g2o::ellipsoid* pEllipsoidForThisObservation = NULL;
            // 2 conditions must meet to start ellipsoid extraction:
            // C1 : the bounding box is not on border
            bool c1 = !is_border;

            // C2 : the groundplane has been estimated successfully
            bool c2 = true; // 对于点模型，一直为真.
            
            // in condition 3, it will not start
            // C3 : under with association mode, and the association is invalid, no need to extract ellipsoids again.
            bool c3 = false;
            if( withAssociation )
            {
                int instance = round(det_vec(7));
                if ( instance < 0 ) c3 = true;  // invalid instance
            }
            
            if( prob_check && c1 && c2 && !c3 ){   

                // 此处提取点云中心即可!
                g2o::ellipsoid e_extracted;
                bool result = mpEllipsoidExtractor->EstimateLocalEllipsoidUsingPointModel(pFrame->frame_img, measurement, label, measurement_prob, pose, mCamera, e_extracted);

                if(result){
                    // 若提取成功
                    pEllipsoidForThisObservation = new g2o::ellipsoid(e_extracted);

                    // 可视化
                    // Visualize estimated ellipsoid
                    g2o::ellipsoid* pObj_world = new g2o::ellipsoid(e_extracted.transform_from(pFrame->cam_pose_Twc));
                    pObj_world->setColor(Vector3d(0.8,0.0,0.0), 1); // Set green color

                    // 第一次添加时清除上一次观测!
                    if(bEllipsoidNotClear)
                    {
                        mpMap->ClearEllipsoidsVisual(); // Clear the Visual Ellipsoids in the map
                        mpMap->ClearBoundingboxes();
                        bEllipsoidNotClear = false;
                    }
                    mpMap->addEllipsoidVisual(pObj_world);
                }

            }

            // 若不成功保持为NULL
            pFrame->mpLocalObjects.push_back(pEllipsoidForThisObservation);
        }
        return;
    }
    // ********************* 
    // 来自 Baseline: TrackingNP.h
    // *********************

    // Update: 2020-6-17 by lzw
    // 支持多支撑平面模式
    void Tracking::UpdateObjectObservationMultiSupportingPlanes(EllipsoidSLAM::Frame *pFrame, bool withAssociation) {
        clock_t time_0_start = clock();

        // 思考: 如何让两个平面提取共用一个 MHPlane 提取.

        // 可以理解为全局重力方向的获取.
        // [1] process MHPlanes estimation
        TaskGroundPlane();
        clock_t time_1_TaskGroundPlane = clock();

        // New task : for Manhattan Planes
        TaskManhattanPlanes(pFrame);
        clock_t time_2_TaskManhattanPlanes = clock();

        // 初始化所有物体的中心点. 存储在 mvpLocalEllipsoids中
        UpdateObjectObservationWithPointModel(pFrame, withAssociation);
        clock_t time_3_UpdateDepthEllipsoidEstimation = clock();

        // [3] Extract Relationship
        TaskRelationshipWithPointModel(pFrame);
        clock_t time_4_TaskRelationship = clock();

        // 目前, 局部平面关联成功的relations 已经存入 pFrame中
        // 还需要考虑局部未成功的，在世界平面中寻找关联!
        // TODO: 先做好一个全局平面的处理模块, 包括前端检测，关联，和后端优化. 放在 Map 中一并管理.
        // 接着，矫正后的位姿必然平面已经与 world 对齐了, 此时直接做分割, 效果估计会好很多.
        // 思考一下全局逻辑，写得清楚一些.!!!


        // [4] Use Relationship To Refine Ellipsoids
        // RefineObjectsWithRelations(pFrame);
        clock_t time_5_RefineObjectsWithRelations = clock();

        GenerateObservationStructure(pFrame);

        // Output running time
        // cout << " -- UpdateObjectObservation Time: " << endl;
        // cout << " --- time_1_TaskGroundPlane: " <<(double)(time_1_TaskGroundPlane - time_0_start) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_2_TaskManhattanPlanes: " <<(double)(time_2_TaskManhattanPlanes - time_1_TaskGroundPlane) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_3_UpdateDepthEllipsoidEstimation: " <<(double)(time_3_UpdateDepthEllipsoidEstimation - time_2_TaskManhattanPlanes) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_4_TaskRelationship: " <<(double)(time_4_TaskRelationship - time_3_UpdateDepthEllipsoidEstimation) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_5_RefineObjectsWithRelations: " <<(double)(time_5_RefineObjectsWithRelations - time_4_TaskRelationship) / CLOCKS_PER_SEC << "s" << endl;        

    }

    void Tracking::UpdateDepthEllipsoidUsingPointModel(EllipsoidSLAM::Frame* pFrame)
    {
        if( !mbDepthEllipsoidOpened ) return;
        
        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int rows = obs_mat.rows();
        Eigen::VectorXd pose = pFrame->cam_pose_Twc.toVector();

        if(pFrame->mpLocalObjects.size() < rows) return;

        int count_point_model = 0;
        int count_ellipsoid_model = 0;
        for(int i=0;i<rows;i++){
            // 下列情况不再进行点模型提取
            // 1) 在前面的过程中，已经产生了有效的椭球体， not NULL, prob_3d > 0.5
            bool c0 = pFrame->mpLocalObjects[i] != NULL && pFrame->mpLocalObjects[i]->prob_3d > 0.5;
            if(c0) 
            {
                count_ellipsoid_model ++;
                continue;
            }

            // ***************
            //  此处执行前有大量条件判断，搬照UpdateDepthEllipsoidEstimation
            // ***************
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate instanceID
            int label = round(det_vec(5));
            double measurement_prob = det_vec(6);

            Eigen::Vector4d measurement = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            // Filter those detections lying on the border.
            bool is_border = calibrateMeasurement(measurement, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            double prob_thresh = Config::Get<double>("Measurement.Probability.Thresh");
            bool prob_check = (measurement_prob > prob_thresh);

            // 2 conditions must meet to start ellipsoid extraction:
            // C1 : the bounding box is not on border
            bool c1 = !is_border;

            // C2 : the groundplane has been estimated successfully
            bool c2 = miGroundPlaneState == 2;
            
            // in condition 3, it will not start
            // C3 : under with association mode, and the association is invalid, no need to extract ellipsoids again.
            bool c3 = false;

            // C4 : 物体过滤
            // 部分动态物体，如人类， label=0，将被过滤不考虑
            bool c4 = true;
            std::set<int> viIgnoreLabelLists = {
                0 // Human
            };
            if(viIgnoreLabelLists.find(label) != viIgnoreLabelLists.end())
                c4 = false;

            if( prob_check && c1 && c2 && !c3 && c4 ){   
                // 单帧椭球体估计失败，或 prob_3d 概率太低，激活中心点估计
                // 为考虑三维分割效果，将其投影到图像平面，与bbox做对比；另一种方法：将概率变得更为显著.
                g2o::ellipsoid e_extracted;
                bool result = mpEllipsoidExtractor->EstimateLocalEllipsoidUsingPointModel(pFrame->frame_img, measurement, label, measurement_prob, pose, mCamera, e_extracted);
                if(result){
                    // 可视化
                    // Visualize estimated ellipsoid
                    g2o::ellipsoid* pObj_world = new g2o::ellipsoid(e_extracted.transform_from(pFrame->cam_pose_Twc));
                    pObj_world->setColor(Vector3d(0,1.0,0.0), 1); // Set green color

                    mpMap->addEllipsoidVisual(pObj_world);

                    g2o::ellipsoid* pEllipsoid = new g2o::ellipsoid(e_extracted);
                    pFrame->mpLocalObjects[i] = pEllipsoid;
                    count_point_model++;
                } 
            }
        }

        std::cout << "[Observations of frame] Total Num : " << rows << std::endl;
        std::cout << " - Ellipsoid Model: " << count_ellipsoid_model << std::endl;
        std::cout << " - Point Model : " << count_point_model << std::endl;
        std::cout << " - Invalid : " << rows-count_ellipsoid_model-count_point_model << std::endl;

        return;

    }

    void Tracking::UpdateObjectObservation(EllipsoidSLAM::Frame *pFrame, bool withAssociation) {
        clock_t time_0_start = clock();

        bool use_infer_detection = Config::Get<int>("System.MonocularInfer.Open") > 0;

        // 思考: 如何让两个平面提取共用一个 MHPlane 提取.

        // [0] 刷新可视化
        ClearVisualization();

        // [1] process MHPlanes estimation
        TaskGroundPlane();
        clock_t time_1_TaskGroundPlane = clock();

        // New task : for Manhattan Planes
        // TaskManhattanPlanes(pFrame);
        clock_t time_2_TaskManhattanPlanes = clock();

        // [2] process single-frame ellipsoid estimation
        clock_t time_3_UpdateDepthEllipsoidEstimation, time_4_TaskRelationship, time_5_RefineObjectsWithRelations;
        
        // ZHJD: 不开启单目版本，则只用点云模型
        if(!use_infer_detection){
            UpdateDepthEllipsoidEstimation(pFrame, withAssociation);
            time_3_UpdateDepthEllipsoidEstimation = clock();

            // [3] Extract Relationship
            TaskRelationship(pFrame);
            time_4_TaskRelationship = clock();

            // [4] Use Relationship To Refine Ellipsoids
            // 注意: Refine时必然在第一步可以初始化出有效的物体.
            RefineObjectsWithRelations(pFrame);
            time_5_RefineObjectsWithRelations = clock();

            // [5] 对于第一次提取，Refine提取都失败的，使用点模型
            UpdateDepthEllipsoidUsingPointModel(pFrame);

            GenerateObservationStructure(pFrame);
        }
        // ZHJD：开启单目版本，则使用单目版本
        // [6] 补充调试环节： 测试语义先验对物体的影响
        else
        {
            GenerateObservationStructure(pFrame);   // 注意必须生成 measure 结构才能 Infer

            const string priconfig_path = Config::Get<std::string>("Dataset.Path.PriTable");
            bool bUseInputPri = (priconfig_path.size() > 0);
            InferObjectsWithSemanticPrior(pFrame, false, use_infer_detection);   // 使用 1:1:1 的比例初始化
            // InferObjectsWithSemanticPrior(pFrame, bUseInputPri, use_infer_detection); // 使用PriTable先验初始化，然而存在问题，尚未调试完毕

            GenerateObservationStructure(pFrame);  
        }

        // Output running time
        // cout << " -- UpdateObjectObservation Time: " << endl;
        // cout << " --- time_1_TaskGroundPlane: " <<(double)(time_1_TaskGroundPlane - time_0_start) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_2_TaskManhattanPlanes: " <<(double)(time_2_TaskManhattanPlanes - time_1_TaskGroundPlane) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_3_UpdateDepthEllipsoidEstimation: " <<(double)(time_3_UpdateDepthEllipsoidEstimation - time_2_TaskManhattanPlanes) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_4_TaskRelationship: " <<(double)(time_4_TaskRelationship - time_3_UpdateDepthEllipsoidEstimation) / CLOCKS_PER_SEC << "s" << endl;        
        // cout << " --- time_5_RefineObjectsWithRelations: " <<(double)(time_5_RefineObjectsWithRelations - time_4_TaskRelationship) / CLOCKS_PER_SEC << "s" << endl;        

    }

    void Tracking::GenerateObservationStructure(EllipsoidSLAM::Frame* pFrame)
    {
        // 本存储结构以物体本身观测为索引.
        // pFrame->meas;
        
        Eigen::MatrixXd &obs_mat = pFrame->mmObservations;
        int ob_num = obs_mat.rows();

        // 3d ob
        std::vector<g2o::ellipsoid*> pLocalObjects = pFrame->mpLocalObjects;

        // if(ob_num != pLocalObjects.size()) 
        // {
        //     std::cout << " [Error] 2d observations and 3d observations should have the same size." << std::endl;
        //     return;
        // }

        for( int i = 0; i < ob_num; i++)
        {
            Eigen::VectorXd det_vec = obs_mat.row(i);  // id x1 y1 x2 y2 label rate imageID
            int label = round(det_vec(5));
            Eigen::Vector4d bbox = Eigen::Vector4d(det_vec(1), det_vec(2), det_vec(3), det_vec(4));

            Observation ob_2d;
            ob_2d.label = label;
            ob_2d.bbox = bbox;
            ob_2d.rate = det_vec(6);
            ob_2d.pFrame = pFrame;

            Observation3D ob_3d;
            ob_3d.pFrame = pFrame;
            if(pLocalObjects.size() == ob_num)
                ob_3d.pObj = pLocalObjects[i];
            else 
                ob_3d.pObj = NULL;

            Measurement m;
            m.measure_id = i;
            m.instance_id = -1; // not associated
            m.ob_2d = ob_2d;
            m.ob_3d = ob_3d;
            pFrame->meas.push_back(m);
        }
    }

    bool GeneratePriFromLabel(int label, Pri& pri)
    {
        // 从 Config 中读取参数
        string prefix_d = "PARAM_SEMANTICPRIOR_D_"; prefix_d+=to_string(label);
        string prefix_e = "PARAM_SEMANTICPRIOR_E_"; prefix_e+=to_string(label);

        double d = Config::ReadValue<double>(prefix_d);
        double e = Config::ReadValue<double>(prefix_e);

        if( d>0 && e>0){
            pri = Pri(d,e);
            return true;
        }
        else 
            return false;

    }

    // 新版本则基于bbox生成，不再与RGBD版本有任何关联
    // replace_detection: 是否将推测物体放入 pFrame 中生效。
    void Tracking::InferObjectsWithSemanticPrior(EllipsoidSLAM::Frame* pFrame, bool use_input_pri = true, bool replace_detection = false)
    {
        // 要求有地平面估计再启动
        if(miGroundPlaneState != 2)
        {
            std::cout << "Close Infering, as the groundplane is not set." << std::endl;
            return;
        }

        // ********* 测试1： 所有先验都是 1:1:1 *********
        // // 读取 pri;  调试模式从全局 Config 中读取
        // double d = Config::ReadValue<double>("SemanticPrior.PriorD");
        // double e = Config::ReadValue<double>("SemanticPrior.PriorE");
        double weight = Config::ReadValue<double>("SemanticPrior.Weight");
        // Pri pri(d,e);

        // std::cout << "Begin infering ... " << std::endl;
        // pri.print();

        // 对于帧内每个物体，做推断，并可视化新的物体
        auto& meas = pFrame->meas;
        int meas_num = meas.size();

        if(replace_detection) {
            pFrame->mpLocalObjects.clear();
            pFrame->mpLocalObjects.resize(meas_num);
        }
        for(int i=0;i<meas_num;i++)
        {
            Measurement& m = meas[i];
            Vector4d bbox = m.ob_2d.bbox;

            // Check : 确保该物体类型是在地面之上的
            if(!CheckLabelOnGround(m.ob_2d.label)) continue;

            // Check : 该 bbox 不在边缘
            bool is_border = calibrateMeasurement(bbox, mRows, mCols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));
            if(is_border) continue;

            // 生成Pri

            // 设置Pri是否是读取或是默认
            Pri pri;
            if(use_input_pri)
                pri = mPrifac.CreatePri(m.ob_2d.label);
            else
                pri = Pri(1,1);
            
            // bool result_pri = GeneratePriFromLabel(m.ob_2d.label, pri);
            // // 如果不成功，我们按 1:1:1 计算?
            // if(!result_pri)
            // {
            //     std::cout << "No pri found for : " << m.ob_2d.label << std::endl;
            //     pri = Pri(1,1);
            // }

            std::cout << "Pri for label : " << m.ob_2d.label << std::endl;
            pri.print();

            // RGB_D + Prior
            g2o::plane ground_pl_local = mGroundPlane; ground_pl_local.transform(pFrame->cam_pose_Tcw);

            priorInfer pi(mRows, mCols, mCalib);
            // *********************************
            // 生成一个新的 Initguess
            // *********************************
            g2o::ellipsoid e_init_guess = pi.GenerateInitGuess(bbox, ground_pl_local.param);
            // ------------------------

            bool bUsePriInit = false;   // 未开发完成的功能
            g2o::ellipsoid e_infer_mono_guess;
            if(bUsePriInit){
                e_infer_mono_guess = pi.MonocularInferExpand(e_init_guess, pri, weight, ground_pl_local);

                // DEBUG 可视化 挨个显示所有的椭球体
                // auto vEsForVisual = pi.GetAllPossibleEllipsoids();
                // for(int i=0;i<vEsForVisual.size();i++)
                // {
                //     g2o::ellipsoid* pETemp = new g2o::ellipsoid(vEsForVisual[i].transform_from(pFrame->cam_pose_Twc));
                //     // mpMap->ClearEllipsoidsVisual();
                //     pETemp->setColor(Vector3d(1.0,0,0));
                //     mpMap->addEllipsoidVisual(pETemp);
                //     std::cout << "Visualize " << i << std::endl;
                //     std::cout << "Wait for push ... " << std::endl;

                //     bool bDebug = false;
                //     if(bDebug)
                //         getchar();
                // }
            }
            else 
                e_infer_mono_guess = pi.MonocularInfer(e_init_guess, pri, weight, ground_pl_local);
            // 设置椭球体label, prob
            e_infer_mono_guess.miLabel = m.ob_2d.label;
            e_infer_mono_guess.prob = m.ob_2d.rate; // 暂时设置为 bbox 检测的概率吧
            e_infer_mono_guess.bbox = m.ob_2d.bbox;
            e_infer_mono_guess.prob_3d =  1.0; // 暂定!
            g2o::ellipsoid* pEInfer_mono_guess = new g2o::ellipsoid(e_infer_mono_guess.transform_from(pFrame->cam_pose_Twc));

            Vector3d color_rgb(144,238,144); color_rgb/=255.0;
            if(!use_input_pri) color_rgb = Vector3d(1,0,0); // 默认版本为红色
            pEInfer_mono_guess->setColor(color_rgb);
            mpMap->addEllipsoidObservation(pEInfer_mono_guess); // 可视化
            std::cout << " Before Monocular Infer: " << e_init_guess.toMinimalVector().transpose() << std::endl;
            std::cout << " After Monocular Infer: " << e_infer_mono_guess.toMinimalVector().transpose() << std::endl;

            // DEBUG可视化： 显示一下初始状态的椭球体
            // g2o::ellipsoid* pE_init_guess = new g2o::ellipsoid(e_init_guess.transform_from(pFrame->cam_pose_Twc));
            // pE_init_guess->prob = 1.0;
            // pE_init_guess->setColor(Vector3d(0.1,0,0.1));
            // mpMap->addEllipsoidVisual(pE_init_guess); // 可视化

            // --------- 将结果放到frame中存储
            if(replace_detection)
                pFrame->mpLocalObjects[i] = new g2o::ellipsoid(e_infer_mono_guess);

            // DEBUGING: 调试为何Z轴会发生变化， 先输出在局部坐标系下的两个rotMat
            // std::cout << "InitGuess RotMat in Camera: " << std::endl << e_init_guess.pose.rotation().toRotationMatrix() << std::endl;
            // std::cout << "Infered RotMat in Camera: " << std::endl << e_infer_mono_guess.pose.rotation().toRotationMatrix() << std::endl;
            // std::cout << "GroundPlaneNorma in Camera: " << std::endl << ground_pl_local.normal().head(3).normalized() << std::endl;

            // 可视化bbox的约束平面
            VisualizeConstrainPlanes(e_infer_mono_guess, pFrame->cam_pose_Twc, mpMap); // 中点定在全局坐标系

        }       

        std::cout << "Finish infering for " << meas_num << " objects..." << std::endl;
        return;
    }


    // // 这个是以前的旧版本，初始推测在椭球体中生成
    // void Tracking::InferObjectsWithSemanticPriorFrom3D(EllipsoidSLAM::Frame* pFrame)
    // {
    //     // 读取 pri;  调试模式从全局 Config 中读取
    //     double d = Config::ReadValue<double>("SemanticPrior.PriorD");
    //     double e = Config::ReadValue<double>("SemanticPrior.PriorE");
    //     double weight = Config::ReadValue<double>("SemanticPrior.Weight");
    //     Pri pri(d,e);

    //     std::cout << "Begin infering ... " << std::endl;
    //     pri.print();

    //     // 对于帧内每个物体，做推断，并可视化新的物体
    //     auto pObs = pFrame->mpLocalObjects;
    //     for(auto pOb : pObs)
    //     {
    //         if(pOb==NULL) continue;
    //         if(pOb->bPointModel) continue; // 点模型跳过
    //         g2o::ellipsoid &e = *pOb;
    //         priorInfer pi(mRows, mCols, mCalib);

    //         // RGB_D + Prior
    //         g2o::plane ground_pl_local = mGroundPlane; ground_pl_local.transform(pFrame->cam_pose_Tcw);
    //         // g2o::ellipsoid e_infer = pi.infer(e, pri, weight, ground_pl_local);
    //         // g2o::ellipsoid* pEInfer = new g2o::ellipsoid(e_infer.transform_from(pFrame->cam_pose_Twc));
    //         // pEInfer->setColor(Vector3d(0,0,1));
    //         // mpMap->addEllipsoidVisual(pEInfer); // 可视化

    //         // Monocular + Prior
    //         // g2o::ellipsoid priorInfer::MonocularInfer(g2o::ellipsoid &e, const Pri &pri, double weight, g2o::plane& plane_ground)
    //         std::cout << "Ground Plane Check: " << mGroundPlane.param.transpose() << std::endl;
    //         // g2o::plane ground_pl_local = mGroundPlane; ground_pl_local.transform(pFrame->cam_pose_Tcw);

    //         // 注释掉： 该部分以RGBD生成的椭球体作为推理初始值
    //         // g2o::ellipsoid e_infer_mono = pi.MonocularInfer(e, pri, weight, ground_pl_local);
    //         // g2o::ellipsoid* pEInfer_mono = new g2o::ellipsoid(e_infer_mono.transform_from(pFrame->cam_pose_Twc));
    //         // pEInfer_mono->setColor(Vector3d(1,0,1));
    //         // mpMap->addEllipsoidVisual(pEInfer_mono); // 可视化

    //         // *********************************
    //         // 添加一个新的 Monocular
    //         // *********************************
    //         double dis_sigma = 0.5; // 0.5m
    //         double size_sigma = 0.1;
    //         ellipsoid e_init_guess = e; // 获得估计出椭球体的 rpy; 只是将 x,y,z,a,b,c 都设置为0.
    //         Vector3d norm_camera_to_obj = e.translation().normalized();
    //         e_init_guess.pose.setTranslation(e.translation()-dis_sigma*norm_camera_to_obj);  // 沿着相机到其中心方向平移一个距离 sigma
    //         e_init_guess.scale = Vector3d(size_sigma,size_sigma,size_sigma);
    //         g2o::ellipsoid e_infer_mono_guess = pi.MonocularInfer(e_init_guess, pri, weight, ground_pl_local);
    //         g2o::ellipsoid* pEInfer_mono_guess = new g2o::ellipsoid(e_infer_mono_guess.transform_from(pFrame->cam_pose_Twc));
    //         pEInfer_mono_guess->setColor(Vector3d(0.5,0,0.5));
    //         mpMap->addEllipsoidVisual(pEInfer_mono_guess); // 可视化
    //         std::cout << " Before Monocular Infer: " << e_init_guess.toMinimalVector().transpose() << std::endl;
    //         std::cout << " After Monocular Infer: " << e_infer_mono_guess.toMinimalVector().transpose() << std::endl;

    //         // DEBUG可视化： 显示一下初始状态的椭球体
    //         g2o::ellipsoid* pE_init_guess = new g2o::ellipsoid(e_init_guess.transform_from(pFrame->cam_pose_Twc));
    //         pE_init_guess->setColor(Vector3d(0.1,0,0.1));
    //         mpMap->addEllipsoidVisual(pE_init_guess); // 可视化

    //     }

    // }

    void Tracking::OpenDepthEllipsoid(){
        mbDepthEllipsoidOpened = true;

        mpEllipsoidExtractor = new EllipsoidExtractor;
        
        // Open visualization during the estimation process
        mpEllipsoidExtractor->OpenVisualization(mpMap);

        // Open symmetry
        if(Config::Get<int>("EllipsoidExtraction.Symmetry.Open") == 1)
            mpEllipsoidExtractor->OpenSymmetry();

        std::cout << std::endl;
        cout << " * Open Single-Frame Ellipsoid Estimation. " << std::endl;
        std::cout << std::endl;
    }

    bool Tracking::isKeyFrameForVisualization()
    {
        static Frame* lastVisualizedFrame;
        if( mvpFrames.size() < 2 ) 
        {
            lastVisualizedFrame = mCurrFrame;
            return true;  
        }

        auto lastPose = lastVisualizedFrame->cam_pose_Twc;
        auto currPose = mCurrFrame->cam_pose_Twc;
        auto diffPose = lastPose.inverse() * currPose;
        Vector6d vec = diffPose.toXYZPRYVector();

        if( (vec.head(3).norm() > 0.4) || (vec.tail(3).norm() > M_PI/180.0*15) )  // Visualization param for camera poses
        {
            lastVisualizedFrame = mCurrFrame;
            return true;
        }
        else
            return false;
    }

    void Tracking::OpenOptimization(){
        mbOpenOptimization = true;
        std::cout << std::endl << "Optimization Opens." <<  std::endl << std::endl ;
    }

    void Tracking::CloseOptimization(){
        mbOpenOptimization = false;
        std::cout << std::endl << "Optimization Closes." <<  std::endl << std::endl ;
    }

    void Tracking::OpenGroundPlaneEstimation(){
        miGroundPlaneState = 1;
        PlaneExtractorParam param;
        param.fx = mK.at<float>(0,0);
        param.fy = mK.at<float>(1,1);
        param.cx = mK.at<float>(0,2);
        param.cy = mK.at<float>(1,2);
        param.scale = Config::Get<double>("Camera.scale");
        pPlaneExtractor = new PlaneExtractor;
        pPlaneExtractor->SetParam(param);

        // Manhattan Task
        miMHPlanesState = 1;
        pPlaneExtractorManhattan = new PlaneExtractorManhattan;
        pPlaneExtractorManhattan->SetParam(param);

        std::cout << " * Open Groundplane Estimation" << std::endl;
        std::cout << std::endl;
    }

    void Tracking::CloseGroundPlaneEstimation(){
        miGroundPlaneState = 0;
        std::cout << std::endl;
        std::cout << " * Close Groundplane Estimation* " << std::endl;
        std::cout << std::endl;
    }

    int Tracking::GetGroundPlaneEstimationState(){
        return miGroundPlaneState;
    }

    void Tracking::ProcessGroundPlaneEstimation()
    {
        cv::Mat depth = mCurrFrame->frame_img;
        g2o::plane groundPlane;
        bool result = pPlaneExtractor->extractGroundPlane(depth, groundPlane);
        g2o::SE3Quat& Twc = mCurrFrame->cam_pose_Twc;  

        // 可视化[所有平面]结果 : 放这里为了让Mannual Check 看见
        auto vPotentialPlanePoints = pPlaneExtractor->GetPoints();
        mpMap->AddPointCloudList("pPlaneExtractor.PlanePoints", vPotentialPlanePoints, Twc, REPLACE_POINT_CLOUD);
        std::cout << " Extract Plane Num : " << vPotentialPlanePoints.size() << std::endl;

        if( result )
        {        
            // 设置世界地平面
            std::cout << " * [Local] Ground plane : " << groundPlane.param.transpose() << std::endl;
            groundPlane.transform(Twc);   // transform to the world coordinate.
            mGroundPlane = groundPlane;
            std::cout << " * Estimate Ground Plane Succeeds: " << mGroundPlane.param.transpose() << std::endl;

            // 可视化该平面 : 为了 Mannual Check.
            mGroundPlane.color = Vector3d(0.0,0.8,0.0); 
            mGroundPlane.InitFinitePlane(Twc.translation(), 10);
            mpMap->addPlane(&mGroundPlane);

            // Active the mannual check of groundplane estimation.
            int active_mannual_groundplane_check = Config::Get<int>("Plane.MannualCheck.Open");
            int key = -1;
            bool open_mannual_check = active_mannual_groundplane_check==1;
            bool result_mannual_check = false;
            if(open_mannual_check)
            {
                std::cout << "Estimate Groundplane Done." << std::endl;
                std::cout << "As Groundplane estimation is a simple implementation, please mannually check its correctness." << std::endl;
                std::cout << "Enter Key \'Y\' to confirm, and any other key to cancel this estimation: " << std::endl;

                key = getchar();
            }

            result_mannual_check = (key == 'Y' || key == 'y');            

            if( !open_mannual_check || (open_mannual_check &&  result_mannual_check) )
            {
                ActivateGroundPlane(mGroundPlane);
            }
            else
            {
                std::cout << " * Cancel this Estimation. " << std::endl;
                miGroundPlaneState = 1;
            }

            // 可视化 : [所有潜在地平面], 从中选择了距离最近的一个
            auto vPotentialGroundplanePoints = pPlaneExtractor->GetPotentialGroundPlanePoints();
            mpMap->AddPointCloudList("pPlaneExtractor.PlanePoints", vPotentialGroundplanePoints, Twc, REPLACE_POINT_CLOUD);
        }
        else
        {
            std::cout << " * Estimate Ground Plane Fails " << std::endl;
        }
    }

    bool Tracking::SavePointCloudMap(const string& path)
    {
        std::cout << "Save pointcloud Map to : " << path << std::endl;
        mpBuilder->saveMap(path);

        return true;
    }

    // This function saves the object history, which stores all the optimized object vector after every new observations.
    void Tracking::RefreshObjectHistory()
    {
        // Object Vector[11]:  optimized_time[1] | Valid/inValid(1/0)[1] | minimal_vec[9] 
        std::map<int, ellipsoid*> pEllipsoidsMapWithInstance = mpMap->GetAllEllipsoidsMap();
        for( auto pairInsPEllipsoid : pEllipsoidsMapWithInstance )
        {
            int instance = pairInsPEllipsoid.first;
            if( mmObjectHistory.find(instance) == mmObjectHistory.end() )  // when the instance has no record in the history
            {
                MatrixXd obHistory; obHistory.resize(0, 11);
                mmObjectHistory.insert(make_pair(instance, obHistory));
            }

            // Add new history
            VectorXd hisVec; hisVec.resize(11);
            assert(mmObjectObservations.find(instance)!=mmObjectObservations.end() && "How does the ellipsoid get into the map without observations?");

            int currentObs = mmObjectObservations[instance].size();
            hisVec[0] = currentObs;  // observation num.
            hisVec[1] = 1;

            Vector9d vec = pairInsPEllipsoid.second->toMinimalVector(); 
            hisVec.tail<9>() = vec;
            
            // Get the observation num of the last history, add new row if the current observation num is newer.
            MatrixXd &obHisMat = mmObjectHistory[instance];
            if( obHisMat.rows() == 0)
            {   
                // Save to the matrix
                addVecToMatirx(obHisMat, hisVec);
            }
            else {
                int lastObNum = round(obHisMat.row(obHisMat.rows()-1)[0]);
                if( lastObNum == currentObs )       // Compare with last observation
                {
                    // Cover it and remain the same
                    obHisMat.row(obHisMat.rows()-1) = hisVec;
                }
                else
                    addVecToMatirx(obHisMat, hisVec);   // Add a new row.
            }
        }
    }

    // Save the object history into a text file.
    void Tracking::SaveObjectHistory(const string& path)
    {
        /*
        *   TotalInstanceNum
        *   instanceID1 historyNum
        *   0 Valid(1/0) minimalVec
        *   1 Valid(1/0) minimalVec
        *   2 Valid(1/0) minimalVec
        *   ...
        *   instanceID2 historyNum
        *   0 Valid(1/0) minimalVec
        *   1 Valid(1/0) minimalVec
        *   ...
        *   
        */ 
        return;     // 暂时取消该功能

        ofstream out(path.c_str());
        int total_num = mmObjectHistory.size();

        out << total_num << std::endl;
        for( auto obPair : mmObjectHistory )
        {
            int instance = obPair.first;
            MatrixXd &hisMat = obPair.second;

            int hisNum = hisMat.rows();
            out << instance << " " << hisNum << std::endl;
            for( int n=0;n<hisNum; n++)
            {
                VectorXd vec = hisMat.row(n);
                int vecNum = vec.rows();
                for( int i=0; i<vecNum; i++){
                    out << vec[i];
                    if(i==vecNum-1)
                        out << std::endl;
                    else
                        out << " ";
                }
            }
        }
        out.close();
        std::cout << "Save object history to " << path << std::endl;
    }

    // 该函数保存每帧提取的曼哈顿平面结果. 一行一个平面参数.
    void Tracking::SaveFrameAndPlaneResult(std::vector<Frame *> & vpFrames)
    {
        string dataset_path = Config::Get<string>("Dataset.Path.Root");
        string dataset_type = Config::Get<string>("Dataset.Type");
        
        string output_dir = "./detection_3d_output/";
        if(dataset_path.size() > 0)
            output_dir = dataset_path + "/" + output_dir;

        string output_dir_relations = "./detection_3d_output/relations/";
        if(dataset_path.size() > 0)
                    output_dir_relations = dataset_path + "/" + output_dir_relations;
        // 文件命名方式 :   plane_xxxx.xxx.txt
        int num_frame = vpFrames.size();
        for(int i=0;i<num_frame;i++)
        {
            Frame* pFrame = vpFrames[i];
            double timestamp = pFrame->timestamp;

            // Relations.
            if(!pFrame->mbSetRelation) continue;

            auto rls = pFrame->relations;
            int num_rl = rls.size();

            int time_stamp_int = round(timestamp);
            string file_name;
            if(dataset_type.find(string("ICL-NUIM"))!=dataset_type.npos)
                file_name = to_string(time_stamp_int); // round 针对 ICL-NUIM 整形存储.
            else 
                file_name = to_string(timestamp);
            string full_file_name = output_dir_relations + file_name + ".txt";

            ofstream fout(full_file_name.c_str());
            for(int n=0;n<num_rl;n++)
            {
                auto rl = rls[n];               
                VectorXd vec_save = rl.SaveToVec();

                // 输出该行
                for(int p=0;p<vec_save.size();p++){
                    fout << vec_save[p];
                    if(p!=vec_save.size()-1)
                        fout << " ";
                    else fout << std::endl;
                }
            }
            fout.close();
        }
        std::cout << "Save " << num_frame << " relations to " << output_dir << std::endl;

        return;
    }

    void Tracking::SaveFrameAndDetectionResult(std::vector<Frame *> & vpFrames, bool save_cplanes)
    {
        string dataset_path = Config::Get<string>("Dataset.Path.Root");
        string dataset_type = Config::Get<string>("Dataset.Type");
        
        string output_dir = "./detection_3d_output/";
        if(dataset_path.size() > 0)
            output_dir = dataset_path + "/" + output_dir;
        // -----
        int num_frame = vpFrames.size();
        for(int i=0;i<num_frame;i++)
        {
            Frame* pFrame = vpFrames[i];
            double timestamp = pFrame->timestamp;

            std::vector<g2o::ellipsoid*> mpLocalObjects = pFrame->mpLocalObjects;
            int num_obj = mpLocalObjects.size();

            int time_stamp_int = round(timestamp);
            string file_name;
            if(dataset_type.find(string("ICL-NUIM"))!=dataset_type.npos)
                file_name = to_string(time_stamp_int); // round 针对 ICL-NUIM 整形存储.
            else 
                file_name = to_string(timestamp);
            string full_file_name = output_dir + file_name + ".txt";

            ofstream fout(full_file_name.c_str());
            for(int n=0;n<num_obj;n++)
            {
                g2o::ellipsoid* pLocalObj = mpLocalObjects[n];
                if(pLocalObj == NULL) continue; 
                
                VectorXd vec_save;
                if(save_cplanes)
                    vec_save = pLocalObj->SaveToVectorWithVecPlanes();
                else
                    vec_save = pLocalObj->SaveToVector();

                // 输出该行
                for(int p=0;p<vec_save.size();p++){
                    fout << vec_save[p];
                    if(p!=vec_save.size()-1)
                        fout << " ";
                    else fout << std::endl;
                }
            }
            fout.close();
        }
        std::cout << "Save " << num_frame << " object detections to " << output_dir << std::endl;
        if(save_cplanes)
            std::cout << "[ ConstrainPlanes were saved. ]" << std::endl;
        else 
            std::cout << "[ ConstrainPlanes were NOT saved. ]" << std::endl;

        return;
    }

    void Tracking::SaveGroundPlane()
    {
        string dataset_path = Config::Get<string>("Dataset.Path.Root");
        string dataset_type = Config::Get<string>("Dataset.Type");
        
        string output_dir = "/detection_3d_output/";

        string full_file_name = dataset_path + output_dir + "/environment/groundplane.txt";
        ofstream fout(full_file_name.c_str());
        
        // int miGroundPlaneState; // 0: Closed  1: estimating 2: estimated 3: set by mannual
        // g2o::plane mGroundPlane;
        if(miGroundPlaneState == 2 )
            fout << mGroundPlane.param.transpose() << std::endl;
        else 
            fout << Vector4d(0,0,0,0).transpose() << std::endl;
        fout.close();

        std::cout << "Save groundplane to " << full_file_name << std::endl;
        
    }

    bool Tracking::NonparamOptimization(const OBJECT_MODEL& model)
    {        
        if(model == QUADRIC_MODEL)
            mpOptimizer->GlobalObjectGraphOptimizationWithPDA(mvpFrames, mpMap, mCalib, mRows, mCols);
        else if(model == POINT_MODEL)
            mpOptimizer->GlobalObjectGraphOptimizationWithPDAPointModel(mvpFrames, mpMap);
        // 该函数以 instance 为索引记录物体历史, 此时Optimization过程id不断变化无效
        // RefreshObjectHistory();  // 5-10日for demo: 存在BUG, 所以关掉它

        // 输出所有帧, 所有局部椭球体的提取结果.
        // SaveFrameAndDetectionResult(mvpFrames, true);   // true: save constrain planes.
        // SaveFrameAndPlaneResult(mvpFrames);
        // SaveGroundPlane();
        return true;
    }

    EllipsoidSLAM::PointCloud* filterCloudAsHeight(EllipsoidSLAM::PointCloud* pCloud, const g2o::plane& mGroundPlane, double dis_thresh)
    {
        std::cout << "Filter Cloud using thresh : " << dis_thresh << std::endl;
        EllipsoidSLAM::PointCloud* pCloudFiltered = new EllipsoidSLAM::PointCloud;
        int num = pCloud->size();
        for(int i=0;i<num;i++)
        {
            PointXYZRGB p = (*pCloud)[i];
            Vector3d center; center << p.x, p.y, p.z;

            double dis = mGroundPlane.distanceToPoint(center);
            if(dis < dis_thresh)
                pCloudFiltered->push_back(p);
        }
        return pCloudFiltered;
    }

    void Tracking::LoadPointcloud(const string& strPcdDir, const string& strPointcloud_name, g2o::SE3Quat Ttrans)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    	pcl::io::loadPCDFile<pcl::PointXYZRGB>(strPcdDir.c_str(), *cloud);

        Matrix4d transform = Ttrans.to_homogeneous_matrix();
        pcl::transformPointCloud (*cloud, *cloud, transform);

        auto pCloud = pclToQuadricPointCloudPtr(cloud);

        // 临时过滤顶部
        double dis_thresh = Config::ReadValue<double>("Visualization.Map.Filter.DisThresh");
        if(dis_thresh > 0){
            auto pCloud_filtered = filterCloudAsHeight(pCloud, mGroundPlane, dis_thresh);
            delete pCloud;
            pCloud = pCloud_filtered;
        }
        mpMap->AddPointCloudList(strPointcloud_name, pCloud);

        return;
    }

    std::vector<Frame*> Tracking::GetAllFrames()
    {
        return mvpFrames;
    }

    void Tracking::Save(std::vector<Frame*>& pFrames)
    {
        int frame_num = pFrames.size();
        mvSavedFramePosesTwc.clear();
        mvSavedFramePosesTwc.resize(frame_num);
        for( int i=0; i<frame_num; i++)
        {
            g2o::SE3Quat pose_wc = pFrames[i]->cam_pose_Twc;
            mvSavedFramePosesTwc[i] = pose_wc;
        }
        return;
    }
    
    void Tracking::Load(std::vector<Frame*>& pFrames)
    {
        if(mvSavedFramePosesTwc.size() != pFrames.size()) 
        {
             std::cout << " No suitable saved poses. " << std::endl;
             return;
        }

        int frame_num = pFrames.size();
        for(int i=0;i<frame_num;i++)
        {
            pFrames[i]->cam_pose_Twc = mvSavedFramePosesTwc[i];  // 直接覆盖
            pFrames[i]->cam_pose_Tcw = pFrames[i]->cam_pose_Twc.inverse();
        }
        return;
    }

    Builder* Tracking::GetBuilder()
    {
        return mpBuilder;
    }

}