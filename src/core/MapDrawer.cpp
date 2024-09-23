#include "include/core/MapDrawer.h"
#include "include/core/ConstrainPlane.h"

#include <pangolin/pangolin.h>
#include <mutex>

#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <GL/glu.h>

namespace EllipsoidSLAM
{
    // draw axis for ellipsoids
    void MapDrawer::drawAxisNormal()
    {
        float length = 2.0;
        
        // x
        glColor3f(1.0,0.0,0.0); // red x
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f, 0.0f);
        glVertex3f(length, 0.0f, 0.0f);
        glEnd();
    
        // y 
        glColor3f(0.0,1.0,0.0); // green y
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f, 0.0f);
        glVertex3f(0.0, length, 0.0f);
    
        glEnd();
    
        // z 
        glColor3f(0.0,0.0,1.0); // blue z
        glBegin(GL_LINES);
        glVertex3f(0.0, 0.0f ,0.0f );
        glVertex3f(0.0, 0.0f ,length );
    
        glEnd();
    }

    MapDrawer::MapDrawer(const string &strSettingPath, Map* pMap):mpMap(pMap)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        mCalib << fx,  0,  cx,
                0,  fy, cy,
                0,      0,     1;

        mbOpenTransform = false;
    }

    void MapDrawer::drawOneBoundingbox(Matrix3Xd& corners, Vector3d& color, double alpha)
    {
        assert(corners.cols()==8 && "Boundingbox must have 8 corners.");
        glPushMatrix();
        glLineWidth(mCameraLineWidth);
        glColor4d(color[0], color[1], color[2], alpha);
        
        glBegin(GL_LINES);

        // draw cube lines. 

        // Version 1: Connect all the points.
        // for(int m=0;m<corners.cols();m++){
        //     for( int n=m+1; n<corners.cols();n++)
        //     {
        //         int m_first = m;
        //         glVertex3f(corners(0,m_first),corners(1,m_first),corners(2,m_first));
        //         int m_next=n;
        //         glVertex3f(corners(0,m_next),corners(1,m_next),corners(2,m_next));
        //     }
        // }

        // Version 2: Standard Cubes
        // corners_body<< 1, 1, -1, -1, 1, 1, -1, -1,
        //                 1, -1, -1, 1, 1, -1, -1, 1,
        //                 -1, -1, -1, -1, 1, 1, 1, 1;
        // Method:  Connect 1,2,3,4;  5,6,7,8; 1-5,2-6,3-7,4-8;
        for( int i = 0; i < 4; i++)
        {
            int m_first = i;
            int m_next = (i+1)%4;
            glVertex3f(corners(0,m_first),corners(1,m_first),corners(2,m_first));
            glVertex3f(corners(0,m_next),corners(1,m_next),corners(2,m_next));
        }
        for( int i = 0; i < 4; i++)
        {
            int m_first = i + 4;
            int m_next = (i+1)%4 + 4;
            glVertex3f(corners(0,m_first),corners(1,m_first),corners(2,m_first));
            glVertex3f(corners(0,m_next),corners(1,m_next),corners(2,m_next));
        }
        for( int i = 0; i < 4; i++)
        {
            int m_first = i;
            int m_next = i + 4;
            glVertex3f(corners(0,m_first),corners(1,m_first),corners(2,m_first));
            glVertex3f(corners(0,m_next),corners(1,m_next),corners(2,m_next));
        }

        glEnd();
        glPopMatrix();
    }

    // draw external cubes.
    bool MapDrawer::drawObjects(double prob_thresh) {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoids();

        std::vector<ellipsoid*> ellipsoidsVisual = mpMap->GetAllEllipsoidsVisual();
        ellipsoids.insert(ellipsoids.end(), ellipsoidsVisual.begin(), ellipsoidsVisual.end());

        std::vector<ellipsoid*> ellipsoidsObs = mpMap->GetObservationEllipsoids();
        ellipsoids.insert(ellipsoids.end(), ellipsoidsObs.begin(), ellipsoidsObs.end());

        std::vector<ellipsoid*> vec_pointmodel_ellipsoids;
        for( size_t i=0; i<ellipsoids.size(); i++)
        {
            g2o::ellipsoid* pE = ellipsoids[i];
            if(pE->prob < prob_thresh) continue;

            // 对于点模型不显示
            Vector3d scale = pE->scale;
            // bool bPointModel = scale[0] < 0.15 && scale[1] < 0.15 && scale[2] < 0.15;
            double size_point = 0.1;
            bool bPointModel = scale[0] == size_point && scale[1] == size_point && scale[2] == size_point;
            if(bPointModel)
            {
                vec_pointmodel_ellipsoids.push_back(pE);
                continue;
            }

            g2o::ellipsoid e_forshow = *pE;
            if(mbOpenTransform)
                e_forshow = e_forshow.transform_from(mTge);

            Eigen::Matrix3Xd corners = e_forshow.compute3D_BoxCorner();
            Vector3d color;
            if(ellipsoids[i]->isColorSet()){
                color = ellipsoids[i]->getColor();
            }
            else
                color = Vector3d(0, 0, 1.0);
            drawOneBoundingbox(corners, color, 1.0);
        }

        drawAllEllipsoidsInVector(vec_pointmodel_ellipsoids);

        return true;
    }

    const char* GetLabelText_MapDrawer(int id)
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
        if(id>=0 && id < 80)
            return coco_classes[id];
        else return "Unknown";
    }

    void MapDrawer::drawLabelTextOfEllipsoids(std::vector<ellipsoid*>& ellipsoids)
    {   
    //  [DSP与Elliposid整合]这部分代码与dsp的结合有问题，因此不运行
        // for( size_t i=0; i<ellipsoids.size(); i++)
        // {
        //     SE3Quat TmwSE3 = ellipsoids[i]->pose.inverse();

        //     if(mbOpenTransform)
        //         TmwSE3 = (mTge * ellipsoids[i]->pose).inverse(); // Tem

        //     Vector3d scale = ellipsoids[i]->scale;

        //     glPushMatrix();

        //     glLineWidth(mCameraLineWidth/3.0);

        //     if(ellipsoids[i]->isColorSet()){
        //         Vector4d color = ellipsoids[i]->getColorWithAlpha();
        //         glColor4d(color(0),color(1),color(2),color(3));
        //     }
        //     else
        //         glColor3f(0.0f,0.0f,1.0f);

        //     pangolin::OpenGlMatrix Twm;   // model to world
        //     SE3ToOpenGLCameraMatrix(TmwSE3, Twm);
        //     glMultMatrixd(Twm.m);  
        //     glScaled(scale[0],scale[1],scale[2]);

        //     // bool bSizeCheck = scale[0] > 0.15 || scale[1] > 0.15 || scale[2] > 0.15;
        //     double size_point = 0.1;
        //     bool bPointModel = scale[0] == size_point && scale[1] == size_point && scale[2] == size_point;
        //     if(!ellipsoids[i]->bPointModel && !bPointModel)
        //         drawAxisNormal();
        //     glPopMatrix();

        //     // draw text for ellipsoids in the map
        //     int label = ellipsoids[i]->miLabel;
        //     // std::cout << " **** [ Debug label : ] " << label << std::endl;
        //     const char* labelText = GetLabelText_MapDrawer(label);
        //     string strLabel(labelText);

        //     // 3d prob
        //     double prob_3d = ellipsoids[i]->prob_3d;
        //     std::stringstream ss2;
        //     ss2 << std::setprecision(2) << prob_3d;
        //     string str_prob_3d = ss2.str();  

        //     string strInstance = " ";
        //     if(ellipsoids[i]->miInstanceID >= 0)
        //     {
        //         strInstance = to_string(ellipsoids[i]->miInstanceID) + " ";
        //     }

        //     string txt = strInstance + strLabel;
            
        //     // add probability
        //     bool bAddProb = false;
        //     if(bAddProb){
        //         double prob = ellipsoids[i]->prob;
        //         std::stringstream ss;
        //         ss << std::setprecision(2) << prob;
        //         string str_prob = ss.str();  

        //         txt += "[" + str_prob +"]";
        //         if( prob_3d > 0 ) txt += "[" + str_prob_3d + "]";
        //     }

        //     // string txt = strLabel;
        //     Eigen::Matrix3Xd corners = ellipsoids[i]->compute3D_BoxCorner();

        //     glPushMatrix(); //  under world coordinate
        //     int rec_id = 4;  // Draw at one of the points of its external rectangle. 4 : 1,1,1
        //     pangolin::GlFont::I().Text( txt ).Draw(corners(0,rec_id), corners(1,rec_id), corners(2,rec_id)); 

        //     // draw text of object scale
        //     bool bDrawScale = false;
        //     if(bDrawScale){
        //         char text_scale[50];
        //         g2o::ellipsoid* pE = ellipsoids[i];
        //         sprintf(text_scale, "%.2fx%.2fx%.2f", pE->scale[0], pE->scale[1], pE->scale[2]);
        //         string txt_scale_str(text_scale);
        //         glColor3f(1.0f, 0.0f,0.0f);
        //         pangolin::GlFont::I().Text( txt_scale_str ).Draw(corners(0,rec_id), corners(1,rec_id), corners(2,rec_id)+0.05); 
        //     }

        //     glPopMatrix();
        // }

        return;        
    }

    // 加入了 transform
    void MapDrawer::drawAllEllipsoidsInVector(std::vector<ellipsoid*>& ellipsoids)
    {
        for( size_t i=0; i<ellipsoids.size(); i++)
        {
            SE3Quat TmwSE3 = ellipsoids[i]->pose.inverse();

            if(mbOpenTransform)
                TmwSE3 = (mTge * ellipsoids[i]->pose).inverse(); // Tem

            Vector3d scale = ellipsoids[i]->scale;

            glPushMatrix();

            glLineWidth(mCameraLineWidth/3.0);

            if(ellipsoids[i]->isColorSet()){
                Vector4d color = ellipsoids[i]->getColorWithAlpha();
                glColor4d(color(0),color(1),color(2),color(3));
            }
            else
                glColor3f(0.0f,0.0f,1.0f);

            GLUquadricObj *pObj;
            pObj = gluNewQuadric();
            gluQuadricDrawStyle(pObj, GLU_LINE);

            pangolin::OpenGlMatrix Twm;   // model to world
            SE3ToOpenGLCameraMatrix(TmwSE3, Twm);
            glMultMatrixd(Twm.m);  
            glScaled(scale[0],scale[1],scale[2]);

            gluSphere(pObj, 1.0, 26, 13); // draw a sphere with radius 1.0, center (0,0,0), slices 26, and stacks 13.

            glPopMatrix();

        }

        return;
    }

    void MapDrawer::drawEllipsoidsLabelText(double prob_thresh, bool show_ellipsoids, bool show_observation)
    {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoids();
        int num_origin = ellipsoids.size();

        if(show_ellipsoids){
            std::vector<ellipsoid*> ellipsoidsVisual = mpMap->GetAllEllipsoidsVisual();
            ellipsoids.insert(ellipsoids.end(), ellipsoidsVisual.begin(), ellipsoidsVisual.end());
        }

        if(show_observation){
            // std::cout << "ellipsoidsVisual.size() = " << ellipsoidsVisual.size() << std::endl;
            std::vector<ellipsoid*> ellipsoidsOb = mpMap->GetObservationEllipsoids();
            ellipsoids.insert(ellipsoids.end(), ellipsoidsOb.begin(), ellipsoidsOb.end());
        }

        // filter those ellipsoids with prob
        std::vector<ellipsoid*> ellipsoids_prob;
        for(auto& pE : ellipsoids)
        {
            if(pE->prob > prob_thresh )
                ellipsoids_prob.push_back(pE);
        }
        
        //  [DSP与Elliposid整合]这部分代码与dsp的结合有问题，因此不运行
        // drawLabelTextOfEllipsoids(ellipsoids_prob);

        return;
    }

    // draw ellipsoids
    bool MapDrawer::drawEllipsoids(double prob_thresh) {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoids();
        int num_origin = ellipsoids.size();

        std::vector<ellipsoid*> ellipsoidsVisual = mpMap->GetAllEllipsoidsVisual();
        ellipsoids.insert(ellipsoids.end(), ellipsoidsVisual.begin(), ellipsoidsVisual.end());

        // filter those ellipsoids with prob
        std::vector<ellipsoid*> ellipsoids_prob;
        for(auto& pE : ellipsoids)
        {
            if(pE->prob > prob_thresh )
                ellipsoids_prob.push_back(pE);
        }
        
        drawAllEllipsoidsInVector(ellipsoids_prob);

        return true;
    }

    bool MapDrawer::drawObservationEllipsoids(double prob_thresh)
    {
        std::vector<ellipsoid*> ellipsoidsObservation = mpMap->GetObservationEllipsoids();

        // filter those ellipsoids with prob
        std::vector<ellipsoid*> ellipsoids_prob;
        for(auto& pE : ellipsoidsObservation)
        {
            if(pE->prob > prob_thresh )
                ellipsoids_prob.push_back(pE);
        }

        drawAllEllipsoidsInVector(ellipsoids_prob);
        return true;
    }

    // draw all the planes
    bool MapDrawer::drawPlanes(int visual_group) {
        std::vector<plane*> planes = mpMap->GetAllPlanes();
        for( size_t i=0; i<planes.size(); i++) {
            g2o::plane* ppl = planes[i];
            if(ppl->miVisualGroup == visual_group)
                drawPlaneWithEquation(ppl);
        }

        return true;
    }

    // from EllipsoidExtractor::calibRotMatAccordingToGroundPlane
    Eigen::Matrix3d calibRotMatAccordingToAxis(Matrix3d& rotMat, const Vector3d& normal){
        // in order to apply a small rotation to align the z axis of the object and the normal vector of the groundplane,
        // we need calculate the rotation axis and its angle.

        // first get the rotation axis
        Vector3d ellipsoid_zAxis = rotMat.col(2);
        Vector3d rot_axis = ellipsoid_zAxis.cross(normal); 
        if(rot_axis.norm()>0)
            rot_axis.normalize();

        // then get the angle between the normal of the groundplane and the z axis of the object
        double norm1 = normal.norm();
        double norm2 = ellipsoid_zAxis.norm();
        double vec_dot = normal.transpose() * ellipsoid_zAxis;
        double cos_theta = vec_dot/norm1/norm2;
        double theta = acos(cos_theta);     

        // generate the rotation vector
        AngleAxisd rot_angleAxis(theta,rot_axis);

        Matrix3d rotMat_calibrated = rot_angleAxis * rotMat;

        return rotMat_calibrated;
    }

    // A sparse version.
    void MapDrawer::drawPlaneWithEquation(plane *p) {
        if( p == NULL ) return;
        Vector3d center;            // 平面上一点!!
        double size;
        
        Vector3d color = p->color;
        Vector3d normal = p->normal(); 
        if(normal.norm()>0)
            normal.normalize();
        if(!p->mbLimited)
        {
            // an infinite plane, us default size
            center = p->SampleNearAnotherPoint(Vector3d(0,0,0));
            size = 10;
        }
        else
        {
            size = p->mdPlaneSize;
            center = p->SampleNearAnotherPoint(p->mvPlaneCenter);
        }
    
        // draw the plane
        Matrix3d rotMat = Matrix3d::Identity();
        // 将其z轴旋转到 normal 方向.
        Matrix3d rotMatCalib = calibRotMatAccordingToAxis(rotMat, normal);

        Vector3d basis_x = rotMatCalib.col(0);
        Vector3d basis_y = rotMatCalib.col(1);

        // const Vector3d v1(center - (basis_x * size) - (basis_y * size));
        // const Vector3d v2(center + (basis_x * size) - (basis_y * size));
        // const Vector3d v3(center + (basis_x * size) + (basis_y * size));
        // const Vector3d v4(center - (basis_x * size) + (basis_y * size));

        // // Draw wireframe plane quadrilateral:

        // // 外轮廓.??
        // drawLine(v1, v2, color, line_width);
        // drawLine(v2, v3, color, line_width);
        // drawLine(v3, v4, color, line_width);
        // drawLine(v4, v1, color, line_width);

        // 绘制内部线条.
        Vector3d point_ld = center - size/2.0 * basis_x - size/2.0 * basis_y;

        double line_width = 2.0;
        double alpha = 0.8;
        int sample_num = 7; // 格子数量
        double sample_dis = size / sample_num;
        for(int i=0;i<sample_num+1;i++)
        {
            // 从起始到结束, 包含起始和结束的等距离sample
            Vector3d v1(point_ld + i*sample_dis*basis_x);
            Vector3d v2(v1 + size*basis_y);
            drawLine(v1, v2, color, line_width, alpha);
        }
        for(int i=0;i<sample_num+1;i++)
        {
            // 从起始到结束, 包含起始和结束的等距离sample
            Vector3d v1(point_ld + i*sample_dis*basis_y);
            Vector3d v2(v1 + size*basis_x);
            drawLine(v1, v2, color, line_width, alpha);
        }

        bool bDrawDirection = true; // 绘制法向量方向
        double direction_length = size / 3;
        if(bDrawDirection)
        {
            Vector3d end_point = center + normal * direction_length;
            drawLine(center, end_point, color/2.0, line_width/1.5, alpha);

            Vector3d end_point2 = end_point - normal * (direction_length / 4);
            drawLine(end_point, end_point2, color*2.0, line_width*2, alpha);// 绘制末端
        }

        return;
    }

    // draw a single plane with dense point cloud filling it.
    void MapDrawer::drawPlaneWithEquationDense(plane *p) {
        if( p == NULL ) return;

        double pieces = 200;
        double ending, starting;

        // sample x and y0
        std::vector<double> x,y,z;
        if( p->mbLimited )      // draw a finite plane.
        {
            pieces = 100;
            
            double area_range = p->mdPlaneSize;
            double step = area_range/pieces;

            // ----- x
            x.clear();
            x.reserve(pieces+2);

            starting = p->mvPlaneCenter[0] - area_range/2;
            ending = p->mvPlaneCenter[0] + area_range/2;

            while(starting <= ending) {
                x.push_back(starting);
                starting += step;
            }

            // ----- y
            y.clear();
            y.reserve(pieces+2);

            starting = p->mvPlaneCenter[1] - area_range/2;
            ending = p->mvPlaneCenter[1] + area_range/2;

            while(starting <= ending) {
                y.push_back(starting);
                starting += step;
            }

            // ----- z
            z.clear();
            z.reserve(pieces+2);

            starting = p->mvPlaneCenter[2] - area_range/2;
            ending = p->mvPlaneCenter[2] + area_range/2;

            while(starting <= ending) {
                z.push_back(starting);
                starting += step;
            }            
        }
        else    // draw an infinite plane, make it big enough
        {
            starting = -5;
            ending = 5;

            x.clear();
            double step = (ending-starting)/pieces;
            x.reserve(pieces+2);
            while(starting <= ending) {
                x.push_back(starting);
                starting += step;
            }
            y=x;
            z=x;
        }
        
        Vector4d param = p->param;
        Vector3d color = p->color;

        glPushMatrix();
        glBegin(GL_POINTS);
        glColor3f(color[0], color[1], color[2]);

        double param_abs_x = std::abs(param[0]);
        double param_abs_y = std::abs(param[1]);
        double param_abs_z = std::abs(param[2]);

        if( param_abs_z > param_abs_x && param_abs_z > param_abs_y ){
            // if the plane is extending toward x axis, use x,y to calculate z.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   Z = (-D-BY-AX)/C
                    double z_  = (-param[3]-param[1]*y[j]-param[0]*x[i])/param[2];

                    glVertex3f(float(x[i]), float(y[j]), float(z_));
                }
            }
        }
        else if( param_abs_x > param_abs_z && param_abs_x > param_abs_y )
        {
            // if the plane is extending toward z axis, use y,z to calculate x.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   X = (-D-BY-CZ)/A
                    double x_  = (-param[3]-param[1]*y[j]-param[2]*z[i])/param[0];

                    glVertex3f(float(x_), float(y[j]), float(z[i]));
                }
            }
        }
        else
        {
            // if the plane is extending toward y axis, use x,z to calculate y.
            for(int i=0; i<pieces; i++)
            {
                for(int j=0; j<pieces; j++)
                {
                    if( i==0 || i==(pieces-1) || j==0 || j == (pieces-1))
                        glColor3f(1.0, 1.0, 0); // highlight the boundary
                    else 
                        glColor3f(color[0], color[1], color[2]);

                    // AX+BY+CZ+D=0    ->   Y = (-D-AX-CZ)/B
                    double y_  = (-param[3]-param[0]*x[j]-param[2]*z[i])/param[1];

                    glVertex3f(float(x[j]), float(y_), float(z[i]));
                }
            }
        }
        
        glEnd();
        glPopMatrix();
    }

    bool MapDrawer::drawGivenCameraState(g2o::SE3Quat* cameraState, const Vector3d& color)
    {
        pangolin::OpenGlMatrix Twc;
        if( cameraState!=NULL )
            SE3ToOpenGLCameraMatrixOrigin(*cameraState, Twc);
        else
        {
            std::cerr << "Can't load camera state." << std::endl;
            Twc.SetIdentity();
        }

        const float &w = mCameraSize*1.5;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m); 
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3d(color[0],color[1],color[2]);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

    bool MapDrawer::drawCameraState() {
        drawGivenCameraState(mpMap->getCameraState(), Vector3d(0,1,0));
        return true;
    }

    // Lines: 每个位置绘制为一个点，并且使用连线连接在一起.

    bool MapDrawer::drawGivenTrajDetail(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color)
    {
        for(int i=0; i<traj.size(); i++)
        {
            g2o::SE3Quat* cameraState = traj[i];        // Twc
            if(!cameraState) continue;
            // Vector3d center = cameraState->translation();    

            // if(mbOpenTransform){
            //     center = (mTge * (*cameraState)).translation();
            // }

            drawGivenCameraState(cameraState, color);
        }

        return true;
    }

    // 添加 transform 的矫正
    bool MapDrawer::drawGivenTrajWithColorLines(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color)
    {
        glPushMatrix();
        glLineWidth(mCameraLineWidth);
        glColor3d(color[0],color[1],color[2]);
        glBegin(GL_LINES);
        for(int i=0; i<traj.size(); i++)
        {
            g2o::SE3Quat* cameraState = traj[i];        // Twc
            if(!cameraState) continue;
            Vector3d center = cameraState->translation();    

            if(mbOpenTransform){
                center = (mTge * (*cameraState)).translation();
            }

        
            glVertex3f(center[0],center[1],center[2]);

            if(i==0) continue;
            glVertex3f(center[0],center[1],center[2]);

        }

        glEnd();
        glPopMatrix();
    }

    // 将每个位置绘制为一个相机模式
    bool MapDrawer::drawGivenTrajWithColor(std::vector<g2o::SE3Quat*>& traj, const Vector3d& color)
    {
        for(int i=0; i<traj.size(); i++)
        {
            g2o::SE3Quat* cameraState = traj[i];        // Twc
            pangolin::OpenGlMatrix Twc;

            if( cameraState!=NULL )
                SE3ToOpenGLCameraMatrixOrigin(*cameraState, Twc);
            else
            {
                std::cerr << "Can't load camera state." << std::endl;
                Twc.SetIdentity();
            }

            const float &w = mCameraSize;
            const float h = w*0.75;
            const float z = w*0.6;

            glPushMatrix();

        #ifdef HAVE_GLES
            glMultMatrixf(Twc.m);  
        #else
            glMultMatrixd(Twc.m);
        #endif

            glLineWidth(mCameraLineWidth);
            glColor3d(color[0],color[1],color[2]);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    std::vector<g2o::SE3Quat*> GetRidOfTimestamp(Trajectory& mapNameTraj)
    {
        int num = mapNameTraj.size();
        std::vector<g2o::SE3Quat*> trajNoStamp; trajNoStamp.resize(num);
        for(int i=0;i<num;i++)
        {
            trajNoStamp[i] = &mapNameTraj[i]->pose;
        }
        return trajNoStamp;
    }

    bool MapDrawer::drawTrajectoryWithName(const string& name)
    {
        Trajectory mapNameTraj = mpMap->getTrajectoryWithName(name);

        // 根据字符计算一组随机颜色
        Vector3d rgb;
        if(name == "OptimizedTrajectory")
            rgb << 0,1.0,0;
        else if(name == "AlignedGroundtruth")
            rgb << 1.0,0,0;
        else{
            int randNum = 0;
            for(int i=0;i<name.size();i++)
                randNum += int(name[i]);
            srand(randNum);
            double r,g,b;
            r = rand()%155 / 255.0;
            g = rand()%155 / 255.0;
            b = rand()%155 / 255.0;
            rgb << r,g,b;
        }

        std::vector<g2o::SE3Quat*> trajForShow = GetRidOfTimestamp(mapNameTraj);

        drawGivenTrajWithColorLines(trajForShow, rgb);
        return true;
    }

    bool MapDrawer::drawTrajectoryDetail()
    {
        std::vector<g2o::SE3Quat*> traj = mpMap->getCameraStateTrajectory();
        drawGivenTrajDetail(traj, Vector3d(0.0,0.0,1.0));
        return true;
    }

    bool MapDrawer::drawTrajectory() {
        std::vector<g2o::SE3Quat*> traj = mpMap->getCameraStateTrajectory();
        drawGivenTrajWithColorLines(traj, Vector3d(0.0,0.0,1.0));
        return true;
    }

    bool MapDrawer::drawPoints() {
        vector<PointXYZRGB*> pPoints = mpMap->GetAllPoints();
        glPushMatrix();

        for(int i=0; i<pPoints.size(); i=i+1)
        {
            PointXYZRGB &p = *(pPoints[i]);
            glPointSize( p.size );
            glBegin(GL_POINTS);
            glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
            glVertex3d(p.x, p.y, p.z);
            glEnd();
        }

        glPointSize( 1 );
        glPopMatrix();

        return true;
    }

    // In : Tcw
    // Out: Twc
    void MapDrawer::SE3ToOpenGLCameraMatrix(g2o::SE3Quat &matInSe3, pangolin::OpenGlMatrix &M)
    {
        // eigen to cv
        Eigen::Matrix4d matEigen = matInSe3.to_homogeneous_matrix();
        cv::Mat matIn;
        eigen2cv(matEigen, matIn);

        if(!matIn.empty())
        {
            cv::Mat Rwc(3,3,CV_64F);
            cv::Mat twc(3,1,CV_64F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = matIn.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*matIn.rowRange(0,3).col(3);
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

    void MapDrawer::eigenMatToOpenGLMat(const Eigen::Matrix4d& matEigen, pangolin::OpenGlMatrix &M)
    {
        cv::Mat matIn;
        eigen2cv(matEigen, matIn);
        if(!matIn.empty())
        {
            cv::Mat Rwc(3,3,CV_64F);
            cv::Mat twc(3,1,CV_64F);
            {
                unique_lock<mutex> lock(mMutexCamera);
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

    // not inverse, keep origin
    void MapDrawer::SE3ToOpenGLCameraMatrixOrigin(g2o::SE3Quat &matInSe3, pangolin::OpenGlMatrix &M)
    {
        // eigen to cv
        Eigen::Matrix4d matEigen = matInSe3.to_homogeneous_matrix();
        eigenMatToOpenGLMat(matEigen, M);
    }

    void MapDrawer::setCalib(Eigen::Matrix3d& calib)
    {
        mCalib = calib;
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
        g2o::SE3Quat *cameraState = mpMap->getCameraState();        // Twc

        if (cameraState != NULL)
            SE3ToOpenGLCameraMatrixOrigin(*cameraState, M);
        else {
            M.SetIdentity();
        }
    }

    void MapDrawer::drawPointCloudLists()
    {
        auto pointLists = mpMap->GetPointCloudList();

        glPushMatrix();

        for(auto pair:pointLists){
            auto pPoints = pair.second;
            if( pPoints == NULL ) continue;
            for(int i=0; i<pPoints->size(); i=i+1)
            {
                PointXYZRGB &p = (*pPoints)[i];
                glPointSize( p.size );
                glBegin(GL_POINTS);
                glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
                glVertex3d(p.x, p.y, p.z);
                glEnd();

            }
        }
        glPointSize( 1 );

        glPopMatrix();
    }

    void MapDrawer::drawPointCloudWithOptions(const std::map<std::string,bool> &options)
    {
        auto pointLists = mpMap->GetPointCloudList();
        if(pointLists.size() < 1) return;
        glPushMatrix();

        for(auto pair:pointLists){
            auto pPoints = pair.second;
            if( pPoints == NULL ) continue;
            
            auto iter = options.find(pair.first);
            if(iter == options.end()) {
                continue;  // not exist
            }
            if(iter->second == false) continue; // menu is closed

            // 拷贝指针指向的点云. 过程中应该锁定地图. (理应考虑对性能的影响)
            PointCloud cloud = mpMap->GetPointCloudInList(pair.first); 
            for(int i=0; i<cloud.size(); i=i+1)
            {
                PointXYZRGB& p = cloud[i];
                glPointSize( p.size );
                glBegin(GL_POINTS);
                glColor3d(p.r/255.0, p.g/255.0, p.b/255.0);
                glVertex3d(p.x, p.y, p.z);
                glEnd();
            }
        }
        glPointSize( 1 );
        glPopMatrix();        
    }

    void MapDrawer::drawBoundingboxes()
    {
        auto vpBoxes = mpMap->GetBoundingboxes();
        for( auto pBox : vpBoxes )
        {
            drawOneBoundingbox(pBox->points, pBox->color, pBox->alpha);
        }
    }

    void MapDrawer::drawConstrainPlanes(double prob_thresh, int type)
    {
        std::vector<ellipsoid*> ellipsoids = mpMap->GetAllEllipsoidsVisual();
        int num = ellipsoids.size();
        for(int n=0; n<num; n++)
        {
            g2o::ellipsoid* pe = ellipsoids[n];
            if(pe->prob < prob_thresh) continue;
            
            // mvCPlanesWorld 存储在世界坐标系, 仅仅用作可视化.
            int pnum = pe->mvCPlanesWorld.size();
            for( size_t i=0; i<pnum; i++) {
                int type_cplane = pe->mvCPlanesWorld[i]->type;

                // // debug check type
                // if(type_cplane!=0 && type_cplane!=1)
                // {
                //     std::cout << "What type is it ? : " << type_cplane << std::endl;
                // }

                if(type_cplane != type ) continue;
                Vector3d color;
                if(pe->mvCPlanesWorld[i]->valid)
                    color = pe->getColor();
                else 
                {
                    if(pe->mvCPlanesWorld[i]->state==1)
                        color = Vector3d(0,0,0);  // 黑色表示在内部,不贡献
                    else if( pe->mvCPlanesWorld[i]->state==2)
                        color = Vector3d(0,0,1.0);    // 蓝色平面表示在外部约束, (生效状态)
                }
                g2o::plane pl(pe->mvCPlanesWorld[i]->pPlane->param.head(4), color);
                pl.InitFinitePlane(pe->pose.translation(), pe->scale.norm()/2.0);
                drawPlaneWithEquation(&pl);
            }
        }
    }

    void MapDrawer::drawArrows()
    {
        std::vector<Arrow> vArs = mpMap->GetArrows();

        for(int i=0;i<vArs.size();i++)
        {
            Arrow& ar = vArs[i];
            Vector3d norm = ar.norm; 
            Vector3d end = ar.center + norm;

            drawLine(ar.center, end, ar.color, mCameraLineWidth * 10, 0.8);
        }

    }

    void MapDrawer::drawLine(const Vector3d& start, const Vector3d& end, const Vector3d& color, double width, double alpha)
    {
        glLineWidth(width);

        glPushMatrix();

        glColor4d(color[0], color[1], color[2], alpha);
        // 先tm画很粗的 Line 吧.
        glBegin(GL_LINES);
        // opengl 画箭头.
        glVertex3d(start[0], start[1], start[2]);
        glVertex3d(end[0], end[1], end[2]);
        glEnd();

        glPopMatrix();
    }

    void MapDrawer::SetTransformTge(const g2o::SE3Quat& Tge)
    {
        mTge = Tge;
        mbOpenTransform = true;
    }
}
