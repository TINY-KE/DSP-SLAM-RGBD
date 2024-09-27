/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{

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



} //namespace ORB_SLAM
