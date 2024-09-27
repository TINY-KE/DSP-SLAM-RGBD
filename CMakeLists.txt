cmake_policy(SET CMP0048 NEW)
project("DSP-SLAM")
cmake_minimum_required(VERSION 3.2)

# set(OpenCV_DIR /home/robotlab/work/DSP-SLAM/Thirdparty/opencv/build)
# set(Eigen3_DIR /home/robotlab/work/DSP-SLAM/Thirdparty/eigen/install/share/eigen3/cmake)
# set(Pangolin_DIR /home/robotlab/work/DSP-SLAM/Thirdparty/Pangolin/build)
# set(PCL_DIR  /home/robotlab/thirdparty_install/pcl-1.13/share/pcl-1.13)
# set(CONDA_ENV "/home/robotlab/anaconda3/envs/dsp-slam")
# set(PYTHON_LIBRARIES ${CONDA_ENV}/lib/libpython3.7m.so)  # Set Python libraries
# set(PYTHON_INCLUDE_DIRS ${CONDA_ENV}/include/python3.7m)   # Set Python include directories
# set(PYTHON_EXECUTABLE ${CONDA_ENV}/bin/python3.7)   # Set Python executable

# c++ version
set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# SET(CMAKE_BUILD_TYPE Debug)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# find_package(OpenCV 3.2 REQUIRED)
find_package(OpenCV 3.4.1 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(/usr/include/vtk-7.1)
set(PCL_DIR  /home/robotlab/thirdparty_install/pcl-1.13/share/pcl-1.13)
FIND_PACKAGE( PCL REQUIRED )


include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/core
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
        ${EIGEN3_INCLUDE_DIR}
                
        ${PCL_INCLUDE_DIRS} 

        ${Pangolin_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}

        ${OpenCV_INCLUDE_DIRS}
        /home/robotlab/anaconda3/envs/dsp-slam/include/python3.7m
)


# You need to set to your own Python paths
add_subdirectory(pybind11)


# 生成elliposid的库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(utils SHARED
        src/utils/matrix_utils.cpp
        src/utils/dataprocess_utils.cpp
        )
target_link_libraries(utils
${OpenCV_LIBS}
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(ELLIPOSID_LIB SHARED
        src/core/Ellipsoid.cpp
        # src/core/Map.cpp
        # src/core/MapDrawer.cpp
        # src/core/Viewer.cpp
        # src/core/Ini/tializer.cpp
        src/core/Geometry.cpp
        # src/core/System.cpp
        # src/core/Tracking.cpp
        # src/core/FrameDrawer.cpp
        # src/core/Optimizer.cpp
        # src/core/Frame.cpp
        src/core/Plane.cpp
        # src/core/DataAssociation.cpp
        src/core/BasicEllipsoidEdges.cpp
        src/core/NonparamDA.cpp
        src/core/SupportingPlane.cpp
        src/core/ConstrainPlane.cpp
        # src/core/SemanticLabel.cpp
        src/core/PriorInfer.cpp
        )

target_link_libraries(ELLIPOSID_LIB
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${Pangolin_LIBRARIES}
        ${OPENGL_LIBRARIES} 
        ${GLUT_LIBRARIES}
        utils
        Config
        symmetry
        EllipsoidExtractor
        PlaneExtractor
        dense_builder
        Polygon
        Relationship
        ) 
# add modules
add_subdirectory(src/tum_rgbd)
add_subdirectory(src/dense_builder)
add_subdirectory(src/symmetry)
add_subdirectory(src/config)
add_subdirectory(src/pca)
add_subdirectory(src/plane)
add_subdirectory(src/Polygon)
add_subdirectory(src/Relationship)







# 生成DSPSLAM的库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/System_util.cc
        src/Tracking.cc
        src/Tracking_util.cc
        src/Tracking_ellipsoid.cc  #张嘉东 生成椭球体
        src/LocalMapping.cc
        src/LocalMapping_util.cc
        src/LoopClosing.cc
        src/LoopClosing_util.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/MapPoint_util.cc
        src/KeyFrame.cc
        src/KeyFrame_util.cc
        src/Map.cc
        src/Map_util.cc
        src/MapDrawer.cc
        src/MapDrawer_ellipsoid.cc
        src/Optimizer.cc
        src/Optimizer_util.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/ObjectDetection.cc
        src/MapObject.cc
        src/ObjectDrawer.cc
        src/ObjectRenderer.cc

        # [整合]
        src/core/Plane.cpp

        # src/core/PriorInfer.cpp
        # src/core/Ellipsoid.cpp
        # src/core/BasicEllipsoidEdges.cpp
        # src/core/Geometry.cpp
        # src/core/ConstrainPlane.cpp
        )

target_link_libraries(${PROJECT_NAME}
        pybind11::embed
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${PCL_LIBRARY_DIRS}
        # ELLIPOSID_LIB
        Config
        # PlaneExtractor
        # EllipsoidExtractor
        # Polygon
        )

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# add_executable(dsp_slam dsp_slam.cc)
# target_link_libraries(dsp_slam ${PROJECT_NAME} /usr/lib/x86_64-linux-gnu/libtiff.so.5)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam_mono dsp_slam_mono.cc)
target_link_libraries(dsp_slam_mono ${PROJECT_NAME} /usr/lib/x86_64-linux-gnu/libtiff.so.5)