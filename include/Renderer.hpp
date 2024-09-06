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


#pragma once

#include <iostream>
#include <pangolin/pangolin.h>

constexpr const char* shader_vert_pinhole =
        R"glsl(
#version 330 core

layout(location = 0) in vec3 vertex;

out vec3 vs_xyz_object;
out float vs_depth;
out vec3 vs_rgb_object;

uniform vec3 object_color;
uniform mat4 MV;  // model view matrix
uniform mat4 M;   // model matrix
uniform mat4 P;   // projection matrix

void main(){
    vec4 p_cam = MV * vec4(vertex, 1);
    gl_Position = P * p_cam;
    vs_xyz_object = p_cam.xyz; // vertex coordinate under camera frame
    vs_depth = p_cam.z;
    vs_rgb_object = object_color;
}
)glsl";


constexpr const char* shader_geom =
        R"glsl(
#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec3 vs_rgb_object[];
in vec3 vs_xyz_object[];
in float vs_depth[];
out vec3 gs_rgb_object;
out vec3 gs_xyz_object;
out float gs_depth;
out vec3 gs_normal;

void main()
{
    vec3 a = (vs_xyz_object[2] - vs_xyz_object[0]);
    vec3 b = (vs_xyz_object[1] - vs_xyz_object[0]);
    vec3 n = normalize(cross(a, b));
    for(int i=0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        gs_normal = n;
        gs_rgb_object = vs_rgb_object[i];
        gs_xyz_object = vs_xyz_object[i];
        gs_depth = vs_depth[i];

        EmitVertex();
    }
    EndPrimitive();
}
)glsl";

constexpr const char* shader_frag =
        R"glsl(
#version 330 core

#extension GL_NV_gpu_shader5 : enable

#define vColor gs_rgb_object
#define vPos gs_xyz_object
#define n gs_normal

in vec3 vColor;
in vec3 vPos;
in vec3 n; // normal vector under camera frame

vec3 lightPos = vec3(0.0, -1000.0, -1000.0);

void main()
{
    vec4 diffuse = vec4(0.0);
    vec4 specular = vec4(0.0);

    // ambient term
    vec4 ambient = vec4(0.5 * vColor, 1);
    // diffuse color
    vec4 kd = vec4(vColor, 1.0);
    // specular color
    vec4 ks = vec4(1.0, 1.0, 1.0, 1.0);
    // diffuse term
    vec3 lightDir = normalize(lightPos - vPos);
    float NdotL = max(dot(lightDir, n), 0.0);
    diffuse = kd * NdotL;

    // specular term
    vec3 rVector = normalize(2.0 * n * dot(n, lightDir) - lightDir);
    vec3 viewVector = normalize(-vPos);
    float RdotV = dot(rVector, viewVector);
    if (RdotV > 0.0)
        specular = ks * pow(RdotV, 16);

    gl_FragColor = ambient + 0.5 * diffuse + 0.8 * specular;
}
)glsl";

/**
 * @brief The XYZMRenderer struct
 * Abstracts rendering of object space (xyz) coordinates.
 */
class Renderer
{
public:

    int viewport_backup[4];
    // 用于设置视口的位置和大小。
    // 其中，viewport_backup数组存储了之前保存的视口参数


    // 这段代码声明了几个变量，它们都是属于Pangolin库中的OpenGL封装类，用于在OpenGL中进行渲染和图形处理。
    // OpenGL着色器程序对象，用于管理顶点着色器、几何着色器和片段着色器的链接和使用。
    pangolin::GlSlProgram program;

    // 这是一个OpenGL帧缓冲对象，用于管理渲染到纹理或渲染缓冲区的操作。
    pangolin::GlFramebuffer framebuffer;

    // OpenGL渲染缓冲对象，通常用于存储深度信息，即深度缓冲。
    pangolin::GlRenderBuffer zbuffer;

    // OpenGL纹理对象，用于存储RGB颜色信息的纹理数据。
    pangolin::GlTexture texture_object_rgb;

    // 【无用】OpenGL纹理对象，用于存储深度信息的纹理数据。 
    pangolin::GlTexture texture_object_depth;

    // 【无用】OpenGL纹理对象，用于存储法线信息的纹理数据。
    pangolin::GlTexture texture_object_normals;

    // 【无用】OpenGL纹理对象，用于存储遮罩信息的纹理数据。 
    pangolin::GlTexture texture_object_mask;

public:
    // 默认构造函数，调用另一个构造函数Renderer(0, 0)。
    Renderer(){
        Renderer(0, 0);
    }
    
    // Renderer(size_t w, size_t h)：带参数的构造函数，初始化渲染器，包括设置着色器和根据传入的宽度和高度创建帧缓冲。
    Renderer(size_t w, size_t h)
    {
        if(pangolin::ShouldQuit())
        {
            throw std::runtime_error("Initializing renderer failed: No context available");
        }
        SetupShader();

        if(w > 0 && h > 0)
        {
            SetupFramebuffer(w, h);
        }
    }

    // Bind()：绑定渲染器，包括绑定帧缓冲、设置视口、启用深度测试和绑定着色器程序。 
    void Bind()
    {
        if(framebuffer.fbid)
        {
            glGetIntegerv(GL_VIEWPORT, viewport_backup);
            framebuffer.Bind();
            glViewport(0, 0, zbuffer.width, zbuffer.height);
        }

        glEnable(GL_DEPTH_TEST);
        program.Bind();
    }

    // Clear()：清除渲染器的颜色缓冲和深度缓冲。
    void Clear()
    {
        if(!framebuffer.fbid)
        {
            throw std::runtime_error("Renderer::Clear() is only available when managing its own framebuffer");
        }
        framebuffer.Bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        framebuffer.Unbind();
    }

    // SetUniformColor(float r, float g, float b)：设置uniform变量object_color的值。
    void SetUniformColor(float r, float g, float b)
    {
        program.SetUniform("object_color", r, g, b);
    }

    void SetUniformMaskID(unsigned char id){
        id = static_cast<GLuint>('t');;
        std::cout<<"RenderId 3： "<<id<<std::endl;
        glUniform1ui(program.GetUniformHandle("id"), id);
    }

    void SetUniformPinholeProjection(const pangolin::OpenGlMatrix& camera_matrix)
    {
        program.SetUniform("P", camera_matrix);
    }

    void SetUniformModelView(const pangolin::OpenGlMatrix& mv)
    {
        program.SetUniform("MV", mv);
    }

    void SetUniformModel(const pangolin::OpenGlMatrix& m)
    {
        program.SetUniform("M", m);
    }

    void SetUniformPinhole(size_t w, size_t h, double fx, double fy, double cx, double cy, const Eigen::Matrix4f& T_co, float near=0.01f, float far=100.0f)
    {
        pangolin::OpenGlMatrix projection_matrix;
        if(framebuffer.fbid)
        {
            projection_matrix = pangolin::ProjectionMatrixRDF_BottomLeft(GLint(w),
                                                                       GLint(h),
                                                                       pangolin::GLprecision(fx),
                                                                       pangolin::GLprecision(fy),
                                                                       pangolin::GLprecision(cx),
                                                                       pangolin::GLprecision(cy),
                                                                       near,
                                                                       far);
        }
        else
        {
            projection_matrix = pangolin::ProjectionMatrixRDF_TopLeft(GLint(w),
                                                                       GLint(h),
                                                                       pangolin::GLprecision(fx),
                                                                       pangolin::GLprecision(fy),
                                                                       pangolin::GLprecision(cx),
                                                                       pangolin::GLprecision(cy),
                                                                       near,
                                                                       far);
        }
        SetUniformPinholeProjection(projection_matrix);
        SetUniformModelView(T_co);
    }

    void Unbind()
    {
        program.Unbind();
        if(framebuffer.fbid)
        {
            framebuffer.Unbind();
            glViewport(viewport_backup[0], viewport_backup[1], viewport_backup[2], viewport_backup[3]);
        }
    }

    void DownloadColor(void* ptr_color)
    {
        if(!framebuffer.fbid || !texture_object_rgb.tid)
        {
            throw std::runtime_error("Renderer::DownloadColor() is only available when managing its own framebuffer and initialized with with_rgb=true");
        }
        texture_object_rgb.Download(ptr_color, GL_RGB, GL_FLOAT);
    }

    inline size_t GetWidth() const{
        if(!framebuffer.fbid){
            throw std::runtime_error("XYZMRenderer::GetWidth() is only available when managing its own framebuffer");
        }
        return zbuffer.width;
    }

    inline size_t GetHeight() const {
        if(!framebuffer.fbid){
            throw std::runtime_error("XYZMRenderer::GetHeight() is only available when managing its own framebuffer");
        }
        return zbuffer.height;
    }


protected:

    void SetupShader()
    {
        std::map<std::string,std::string> defines;

        program.AddShader(pangolin::GlSlVertexShader, shader_vert_pinhole, defines);
        program.AddShader(pangolin::GlSlGeometryShader, shader_geom, defines);
        program.AddShader(pangolin::GlSlFragmentShader, shader_frag, defines);
        program.Link();
        program.Bind();
        program.Unbind();
    }

    void SetupFramebuffer(size_t w, size_t h)
    {
        // rendering buffer
        zbuffer.Reinitialise(GLint(w), GLint(h), GL_DEPTH_COMPONENT32);
        texture_object_rgb.Reinitialise(GLint(w), GLint(h), GL_RGB32F, true, 0, GL_RGB, GL_FLOAT);
        framebuffer.AttachColour(texture_object_rgb);
        framebuffer.AttachDepth(zbuffer);
    }
};
