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

#include "ObjectDrawer.h"

namespace ORB_SLAM2
{

ObjectDrawer::ObjectDrawer(Map *pMap, MapDrawer *pMapDrawer, const string &strSettingPath) : mpMap(pMap), mpMapDrawer(pMapDrawer)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    mViewpointF = fSettings["Viewer.ViewpointF"];
    mvObjectColors.push_back(std::tuple<float, float, float>({230. / 255., 0., 0.}));	 // red  0
    mvObjectColors.push_back(std::tuple<float, float, float>({60. / 255., 180. / 255., 75. / 255.}));   // green  1
    mvObjectColors.push_back(std::tuple<float, float, float>({0., 0., 255. / 255.}));	 // blue  2
    mvObjectColors.push_back(std::tuple<float, float, float>({255. / 255., 0, 255. / 255.}));   // Magenta  3
    mvObjectColors.push_back(std::tuple<float, float, float>({255. / 255., 165. / 255., 0}));   // orange 4
    mvObjectColors.push_back(std::tuple<float, float, float>({128. / 255., 0, 128. / 255.}));   //purple 5
    mvObjectColors.push_back(std::tuple<float, float, float>({0., 255. / 255., 255. / 255.}));   //cyan 6
    mvObjectColors.push_back(std::tuple<float, float, float>({210. / 255., 245. / 255., 60. / 255.}));  //lime  7
    mvObjectColors.push_back(std::tuple<float, float, float>({250. / 255., 190. / 255., 190. / 255.})); //pink  8
    mvObjectColors.push_back(std::tuple<float, float, float>({0., 128. / 255., 128. / 255.}));   //Teal  9
    SE3Tcw = Eigen::Matrix4f::Identity();
    SE3TcwFollow = Eigen::Matrix4f::Identity();
}

void ObjectDrawer::SetRenderer(ObjectRenderer *pRenderer)
{
    mpRenderer = pRenderer;
}

void ObjectDrawer::AddDrawerObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexObjects);
    mlNewMapObjects.push_back(pMO);

}

void ObjectDrawer::ProcessNewObjects()
{
    unique_lock<mutex> lock(mMutexObjects);
    auto pMO = mlNewMapObjects.front();
    if (pMO)
    {   
        //从Render Map中获取物体的数量，并加一，从而获得renderId。
        int renderId = (int) mpRenderer->AddObject(pMO->vertices, pMO->faces);   
        pMO->SetRenderId(renderId);
        mlNewMapObjects.pop_front();
    }
}

void ObjectDrawer::DrawNBV(bool bFollow, const Eigen::Matrix4f &Tec)
{
    unique_lock<mutex> lock(mMutexObjects);

    // 地图中所有的物体
    auto mvpMapObjects = mpMap->GetAllMapObjects();

    for (MapObject *pMO : mvpMapObjects)
    {
        if (!pMO)
            continue;
        if (pMO->isBad())
            continue;

        Eigen::Matrix4f Sim3Two = pMO->GetPoseSim3();
        int idx = pMO->GetRenderId();

        if (bFollow) {
            SE3TcwFollow = SE3Tcw;
        }
        if (pMO->GetRenderId() >= 0)
        {

            //NBV
            pMO->compute_NBV();
            const float &w = 3.0;
            const float h = w*0.75;
            const float z = w*0.6;
            cv::Mat Twc = pMO->nbv->getCVMatPose().t();  //zhjd疑问：这里为什么用了转置？？
//            auto T_w_to_nbv = Converter::toMatrix4d(Twc);
//            std::cout << "zhjddebug NBV坐标" << " " << T_w_to_nbv(0, 3) << " " << T_w_to_nbv(1, 3) << " " << T_w_to_nbv(2, 3) << std::endl;
//            auto T_w_to_nbv2 =  pMO->nbv->pose_isometry3d.matrix();
//            std::cout<<"zhjddebug NBV坐标"<<" "<<T_w_to_nbv2(0, 3) << " " << T_w_to_nbv2(1, 3) << " " << T_w_to_nbv2(2, 3)<<std::endl;
//            std::cout<<"zhjddebug NBV坐标"<<" "<<pMO->nbv->centor_x<<" "<<pMO->nbv->centor_y<<" "<<pMO->nbv->centor_z<<std::endl;

            glPushMatrix();
            glMultMatrixf(Twc.ptr<GLfloat>(0));
            glLineWidth(2);
            glColor3f(1.0f,0.0f,0.0f);
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
}

void ObjectDrawer::DrawObjects(bool bFollow, const Eigen::Matrix4f &Tec)
{
    unique_lock<mutex> lock(mMutexObjects);

    // 地图中所有的物体
    auto mvpMapObjects = mpMap->GetAllMapObjects();  

    for (MapObject *pMO : mvpMapObjects)
    {
        if (!pMO)
            continue;
        if (pMO->isBad())
            continue;

        Eigen::Matrix4f Sim3Two = pMO->GetPoseSim3();  
        int idx = pMO->GetRenderId();

        if (bFollow) {
            SE3TcwFollow = SE3Tcw;
        }
        if (pMO->GetRenderId() >= 0)
        {
            // std::cout<<"RenderId 2： "<<idx<<std::endl;
            mpRenderer->Render(pMO->GetRenderId(), 
                        Tec * SE3TcwFollow /* CurrentCameraPose */ * Sim3Two  //这是
                        , mvObjectColors[pMO->GetRenderId() % mvObjectColors.size()]);
            auto objectpose = Tec * SE3TcwFollow /* CurrentCameraPose */ * Sim3Two;
            std::cout<<"zhjddebug 物体坐标"<<" "<<objectpose(0,3)<<" "<<objectpose(1,3)<<" "<<objectpose(2,3)<<std::endl;

        }
        // DrawCuboid(pMO);
    }
}


void ObjectDrawer::DrawCuboid(MapObject *pMO)
{
    const float w = pMO->w / 2;
    const float h = pMO->h / 2;
    const float l = pMO->l / 2;

    glPushMatrix();

    pangolin::OpenGlMatrix Two = Converter::toMatrixPango(pMO->SE3Two);
#ifdef HAVE_GLES
    glMultMatrixf(Two.m);
#else
    glMultMatrixd(Two.m);
#endif

    const float mCuboidLineWidth = 3.0;
    glLineWidth(mCuboidLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);

    glVertex3f(w,h,l);
    glVertex3f(w,-h,l);

    glVertex3f(-w,h,l);
    glVertex3f(-w,-h,l);

    glVertex3f(-w,h,l);
    glVertex3f(w,h,l);

    glVertex3f(-w,-h,l);
    glVertex3f(w,-h,l);

    glVertex3f(w,h,-l);
    glVertex3f(w,-h,-l);

    glVertex3f(-w,h,-l);
    glVertex3f(-w,-h,-l);

    glVertex3f(-w,h,-l);
    glVertex3f(w,h,-l);

    glVertex3f(-w,-h,-l);
    glVertex3f(w,-h,-l);

    glVertex3f(w,h,-l);
    glVertex3f(w,h,l);

    glVertex3f(-w,h,-l);
    glVertex3f(-w,h,l);

    glVertex3f(-w,-h,-l);
    glVertex3f(-w,-h,l);

    glVertex3f(w,-h,-l);
    glVertex3f(w,-h,l);

    glEnd();

    glPopMatrix();
}

void ObjectDrawer::SetCurrentCameraPose(const Eigen::Matrix4f &Tcw)
{
    unique_lock<mutex> lock(mMutexObjects);
    SE3Tcw = Tcw;
}

}

