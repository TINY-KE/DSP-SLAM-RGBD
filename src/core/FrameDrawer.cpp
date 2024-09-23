#include "include/core/FrameDrawer.h"
#include "src/config/Config.h"
#include "utils/dataprocess_utils.h"

using namespace cv;

namespace EllipsoidSLAM
{
    FrameDrawer::FrameDrawer(EllipsoidSLAM::Map *pMap) {

        mpMap = pMap;
        mmRGB = cv::Mat();
        mmDepth = cv::Mat();
    }

    void FrameDrawer::setTracker(EllipsoidSLAM::Tracking *pTracker) {
        mpTracking = pTracker;
    }

    cv::Mat FrameDrawer::drawFrame() {
        Frame * frame = mpTracking->mCurrFrame;
        if( frame == NULL ) return cv::Mat();
        cv::Mat im;
        if(!frame->rgb_img.empty()) // use rgb image if it exists, or use depth image instead.
            im = frame->rgb_img;
        else 
            im = frame->frame_img;

        cv::Mat out = drawFrameOnImage(im);

        mmRGB = out.clone();

        return mmRGB;
    }

    cv::Mat FrameDrawer::drawDepthFrame() {
        Frame * frame = mpTracking->mCurrFrame;
        cv::Mat I = frame->frame_img;   // U16C1 , ushort
        cv::Mat im,R,G,B;

        double min;
        double max;
        cv::minMaxIdx(I, &min, &max);

        I.convertTo(im,CV_8UC1, 255 / (max-min), -min/(max-min)*255); 

        Vector3d color1(51,25,0);
        Vector3d color2(255,229,204);
        // Vector3d color1(255,20,0);
        // Vector3d color2(0,20,255);

        Vector3d scale_color = (color2-color1) / 255.0;
        // r = color1 + value/255*(color2-color1)
        // (255-51)/255   51
        // 220-25

        // ********************************************
        // Vector3d color1(0,204,102);
        // Vector3d color2(255,204,102);
        // double depth_thresh = 8;
        // double scale = Config::Get<double>("Camera.scale"); 

        // Vector3d scale_param = (color2-color1) / scale / depth_thresh;
        // // I / scale / depth_thresh   (0-1)   * (color2-color1) + color 1

        // I.convertTo(B, CV_8UC1, scale_param[0], color1[0]);
        // I.convertTo(G, CV_8UC1, scale_param[1], color1[1]);
        // I.convertTo(R, CV_8UC1, scale_param[2], color1[2]);
        // *********************************************

        im.convertTo(R, CV_8UC1, scale_color[0], color1[0]);
        im.convertTo(G, CV_8UC1, scale_color[1], color1[1]);
        im.convertTo(B, CV_8UC1, scale_color[2], color1[2]);

        std::vector<cv::Mat> array_to_merge;
        array_to_merge.push_back(B);
        array_to_merge.push_back(G);
        array_to_merge.push_back(R);
        cv::merge(array_to_merge, im);

        cv::Mat out = drawObservationOnImage(im, false);

        mmDepth = out.clone();

        return mmDepth;
    }

    cv::Mat drawPointCloudOnImage(cv::Mat& im, PointCloud& cloud, const Vector3d& color, g2o::SE3Quat& trans, Matrix3d& calib)
    {
        cv::Mat out = im.clone();
        if(cloud.size()==0) return out;
        PointCloud* pCloud_local = transformPointCloud(&cloud, &trans);

        // 局部点云投影到二维坐标
        int num = pCloud_local->size();
        for( int i=0;i<num;i++)
        {
            PointXYZRGB& p = (*pCloud_local)[i];
            // Vector4d xyz_homo; xyz_homo << p.x, p.y, p.z, 1;
            Vector3d xyz; xyz << p.x, p.y, p.z;
            Vector3d uv_homo = calib * xyz;
            Vector2d uv = uv_homo.head(2) / uv_homo[2];

            int x = round(uv[0]);
            int y = round(uv[1]);

            // 绘制到图像上
            if(color[0] >= 0)
                out.at<cv::Vec3b>(y,x) = cv::Vec3b(color[2]*255, color[1]*255, color[0]*255);
            else 
                out.at<cv::Vec3b>(y,x) = cv::Vec3b(p.b, p.g, p.r);
        }

        return out;
    }

    cv::Mat FrameDrawer::drawDepthFrameWithVisualPoints() {
        // cv::Mat depthMat = mmDepth;
        cv::Mat depthMat = drawDepthFrame();

        Matrix3d calib = mpTracking->mCalib;
        g2o::SE3Quat campose_cw = mpTracking->mCurrFrame->cam_pose_Tcw;

        // ----- 绘制投影的点云
        // 得到将要绘制的点云
        PointCloud cloud = mpMap->GetPointCloudInList("EllipsoidExtractor.EuclideanFiltered");
        PointCloud cloud_plane = mpMap->GetPointCloudInList("pPlaneExtractorManhattan.MH-Plane Points");
        // 要求先变换回局部坐标系, 然后投影
        cv::Mat out = drawPointCloudOnImage(depthMat, cloud, Vector3d(0,0,1.0), campose_cw, calib);
        out = drawPointCloudOnImage(out, cloud_plane, Vector3d(-1,0,0), campose_cw, calib); // -1 use own color
        
        mmDepth = out;
        return out;
    }

    cv::Mat FrameDrawer::drawProjectionOnImage(cv::Mat &im) {
        std::map<int, ellipsoid*> pEllipsoidsMapWithLabel = mpMap->GetAllEllipsoidsMap();

        // draw projected bounding boxes of the ellipsoids in the map
        cv::Mat imageProj = im.clone();
        for(auto iter=pEllipsoidsMapWithLabel.begin(); iter!=pEllipsoidsMapWithLabel.end();iter++)
        {
            ellipsoid* e = iter->second;

            // check whether it could be seen
            if( e->CheckObservability(mpTracking->mCurrFrame->cam_pose_Tcw) )
            {
                Vector4d rect = e->getBoundingBoxFromProjection(mpTracking->mCurrFrame->cam_pose_Tcw, mpTracking->mCalib); // center, width, height
                cv::rectangle(imageProj, cv::Rect(cv::Point(rect[0],rect[1]),cv::Point(rect[2],rect[3])), cv::Scalar(0,0,255), 4);
            }
        }

        return imageProj.clone();
    }

    cv::Mat FrameDrawer::drawVisualEllipsoidsProjectionOnImage(cv::Mat &im) {
        std::vector<g2o::ellipsoid*> vElps = mpMap->GetAllEllipsoidsVisual();

        // draw projected bounding boxes of the ellipsoids in the map
        cv::Mat imageProj = im.clone();
        for(auto iter=vElps.begin(); iter!=vElps.end();iter++)
        {
            ellipsoid* e = *iter;
            if(e==NULL) continue;

            // Check Prob!
            double config_prob_thresh = Config::ReadValue<double>("Dynamic.Optimizer.EllipsoidProbThresh");
            if(e->prob < config_prob_thresh) continue;

            // check whether it could be seen
            if( e->CheckObservability(mpTracking->mCurrFrame->cam_pose_Tcw) )
            {
                // draw rectangle
                // Vector4d rect = e->getBoundingBoxFromProjection(mpTracking->mCurrFrame->cam_pose_Tcw, mpTracking->mCalib); // center, width, height
                // cv::rectangle(imageProj, cv::Rect(cv::Point(rect[0],rect[1]),cv::Point(rect[2],rect[3])), cv::Scalar(255,0,255), 2);
                
                // draw ellipse
                Vector5d ellipse = e->projectOntoImageEllipse(mpTracking->mCurrFrame->cam_pose_Tcw, mpTracking->mCalib);
                Vector3d color = e->getColor() * 255;
                cv::RotatedRect rotbox2(cv::Point2f(ellipse[0],ellipse[1]), cv::Size2f(ellipse[3]*2,ellipse[4]*2), ellipse[2]/M_PI*180);
                try
                {
                    cv::ellipse(imageProj, rotbox2, cv::Scalar(color[2], color[1], color[0]), 3);
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
            }
        }

        return imageProj.clone();
    }

    void draw3DBoundingBox(Matrix2Xd& corners, cv::Mat& im)
    {
        // Version 2: Standard Cubes
        // corners_body<< 1, 1, -1, -1, 1, 1, -1, -1,
        //                 1, -1, -1, 1, 1, -1, -1, 1,
        //                 -1, -1, -1, -1, 1, 1, 1, 1;
        // Method:  Connect 1,2,3,4;  5,6,7,8; 1-5,2-6,3-7,4-8;
        int thickness = 2;
        cv::Scalar color = cv::Scalar(255,0,0);
        for( int i = 0; i < 4; i++)
        {
            int m_first = i;
            int m_next = (i+1)%4;
            cv::Point p1(corners(0,m_first),corners(1,m_first));
            cv::Point p2(corners(0,m_next),corners(1,m_next));
            cv::line(im, p1, p2, color, thickness);
        }
        for( int i = 0; i < 4; i++)
        {
            int m_first = i + 4;
            int m_next = (i+1)%4 + 4;
            cv::Point p1(corners(0,m_first),corners(1,m_first));
            cv::Point p2(corners(0,m_next),corners(1,m_next));
            cv::line(im, p1, p2, color, thickness);
        }
        for( int i = 0; i < 4; i++)
        {
            int m_first = i;
            int m_next = i + 4;
            cv::Point p1(corners(0,m_first),corners(1,m_first));
            cv::Point p2(corners(0,m_next),corners(1,m_next));
            cv::line(im, p1, p2, color, thickness);
        }

        return;
    }

    cv::Mat FrameDrawer::draw3DProjectionOnImage(cv::Mat &im) {
        std::vector<g2o::ellipsoid*> vElps = mpMap->GetAllEllipsoidsVisual();

        // draw projected bounding boxes of the ellipsoids in the map
        cv::Mat imageProj = im.clone();
        for(auto iter=vElps.begin(); iter!=vElps.end();iter++)
        {
            ellipsoid* e = *iter;

            // check whether it could be seen
            if( e->CheckObservability(mpTracking->mCurrFrame->cam_pose_Tcw) )
            {
                Matrix2Xd points = e->projectOntoImageBoxCorner(mpTracking->mCurrFrame->cam_pose_Tcw, mpTracking->mCalib);
                draw3DBoundingBox(points, imageProj);
            }
        }

        return imageProj.clone();
    }

    cv::Mat FrameDrawer::drawFrameOnImage(cv::Mat &in) {
        cv::Mat out = in.clone();
        out = drawObservationOnImage(out, true);
        // out = drawProjectionOnImage(out);
        // out = draw3DProjectionOnImage(out);
        out = drawVisualEllipsoidsProjectionOnImage(out);
        return out.clone();
    }

    const char* GetLabelText(int id)
    {
        static const char *coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train",
        "truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird",
        "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
        "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
        "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","monitor",
        "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

        return coco_classes[id];
    }

    cv::Mat FrameDrawer::drawObservationOnImage(cv::Mat &in, bool draw_text) {
        cv::Mat im = in.clone();

        Frame * frame = mpTracking->mCurrFrame;
        // draw observation
        Eigen::MatrixXd mat_det = frame->mmObservations;
        int obs = mat_det.rows();
        
        int fontFace = CV_FONT_HERSHEY_SIMPLEX;

        double size_scale = in.cols / 960.0;
        double fontScale = 1.3 * size_scale;
        int thickness = 2;
        for(int r=0;r<obs;r++){
            VectorXd vDet = mat_det.row(r);

            Vector4d measure; measure << vDet(1), vDet(2), vDet(3), vDet(4);
            bool is_border = calibrateMeasurement(measure, im.rows, im.cols, Config::Get<int>("Measurement.Border.Pixels"), Config::Get<int>("Measurement.LengthLimit.Pixels"));

            int labelId = int(vDet(5));
            Rect rec(Point(vDet(1), vDet(2)), Point(vDet(3), vDet(4)));

            double prob = vDet(6);
            double prob_thresh = Config::Get<double>("Measurement.Probability.Thresh");
            bool prob_check = (prob > prob_thresh);

            int line_size = 3;
            Scalar color = Scalar(0,255,0);

            // 若该检测在边界，则灰度显示
            if( is_border )
            {
                // line_size = 3;
                color = Scalar(128,128,128);
            }
            
            rectangle(im, rec, color, line_size);

            Point textOrg = Point(vDet(1), vDet(2));
          
            if(draw_text){
                // prob
                std::stringstream ss;
                ss << std::setprecision(2) << prob;
                string str_prob = ss.str();  
                // label
                const char* labelText = GetLabelText(labelId);
                string str_labelText(labelText);
                // total
                string str_id_text_prob = to_string(labelId) + " " + str_labelText + " [" + str_prob + "]";
                int baseline = 0;
                Size textSize = getTextSize(str_id_text_prob, fontFace, fontScale, thickness, &baseline);

                // draw
                rectangle(im, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0, 255, 0), -1);
                putText(im, str_id_text_prob, textOrg, fontFace, fontScale, Scalar(255,0,0), thickness);
            }
            else
            {
                // 如果没有draw text, 则draw id: 即在bboxMat中的位置
                string str_id = to_string(r);
                int baseline = 0;
                Size textSize = getTextSize(str_id, fontFace, fontScale, thickness, &baseline);

                rectangle(im, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0, 255, 0), -1);
                putText(im, str_id, textOrg, fontFace, fontScale, Scalar(255,0,0), thickness);
            }
        }

        return im.clone();
    }

    cv::Mat FrameDrawer::getCurrentFrameImage(){
        return mmRGB;
    }

    cv::Mat FrameDrawer::getCurrentDepthFrameImage(){
        return mmDepth;
    }
}