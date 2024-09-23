cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../core/cmake_modules)

# For Opencv, check environment first
if(NOT DEFINED ENV{OpenCV_DIR})
   message("not defined environment variable: OpenCV_DIR")
   find_package(OpenCV 3.0 QUIET)
   if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.11 REQUIRED)
      if(NOT OpenCV_FOUND)
         message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
      endif()
   endif()
else()
   message("Environment variable OpenCV_DIR: " $ENV{OpenCV_DIR})
   set(OpenCV_DIR $ENV{OpenCV_DIR})
   find_package(OpenCV)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV not found.")
   endif()
endif()

message ( "CV include" ${OpenCV_INCLUDE_DIRS})
message ( "CV LIBS: " ${OpenCV_LIBS})
message ( "LD_LIBRARY_PATH: " ${LD_LIBRARY_PATH})

link_directories("/home/lzw/workspace/libraries/opencv-2.4.10/build/lib")

find_package(Eigen3 REQUIRED)
#find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
#${Pangolin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
)

# find new cv_bridge based on OpenCV 2.x
set(cv_bridge_DIR "/usr/local/include/cv_bridge")
message ( "cv_bridge_DIR: " ${cv_bridge_DIR})

find_package(cv_bridge REQUIRED)
include_directories(${cv_bridge_DIR})

message ( "cv_bridge_LIBRARIES: " ${cv_bridge_LIBRARIES})
message ( "cv_bridge_DIR: " ${cv_bridge_DIR})


# 寻找PCL
FIND_PACKAGE( PCL REQUIRED )
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this in Ubuntu 16.04

# 添加头文件和库文件
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

find_package(Pangolin REQUIRED)

# rosbuild_add_executable(ros_ellipsoid_slam
# src/ros_ellipsoid_slam.cpp
# )
# target_link_libraries(ros_ellipsoid_slam
#    boost_system
#    ${OpenCV_LIBS} 
#    ${PROJECT_SOURCE_DIR}/../../lib/libsymmetry.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libEllipsoidSLAM.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libConfig.so
#    ${Pangolin_LIBRARIES}
#    ${cv_bridge_LIBRARIES}
#    ${PCL_LIBRARIES}
#    ${PROJECT_SOURCE_DIR}/../../lib/libEllipsoidExtractor.so

# )

rosbuild_add_executable(ros_ellipsoid_slam_real
src/ros_ellipsoid_slam_real.cpp
)
target_link_libraries(ros_ellipsoid_slam_real
   boost_system
   ${OpenCV_LIBS} 
   # ${PROJECT_SOURCE_DIR}/../../lib/libsymmetry.so
   ${PROJECT_SOURCE_DIR}/../../../release/libutils.so
   ${PROJECT_SOURCE_DIR}/../../../release/libEllipsoidSLAM.so
   ${PROJECT_SOURCE_DIR}/../../../release/src/config/libConfig.so
   ${Pangolin_LIBRARIES}
   ${cv_bridge_LIBRARIES}
   ${PCL_LIBRARIES}
   # ${PROJECT_SOURCE_DIR}/../../lib/libEllipsoidExtractor.so

)

# rosbuild_add_executable(ros_single_detection_for_real
# src/ros_single_detection_for_real.cpp
# )
# target_link_libraries(ros_single_detection_for_real
#    boost_system
#    ${OpenCV_LIBS} 
#    ${PROJECT_SOURCE_DIR}/../../lib/libsymmetry.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libEllipsoidSLAM.so
#    ${PROJECT_SOURCE_DIR}/../../lib/libConfig.so
#    ${Pangolin_LIBRARIES}
#    ${cv_bridge_LIBRARIES}
#    ${PCL_LIBRARIES}
#    ${PROJECT_SOURCE_DIR}/../../lib/libEllipsoidExtractor.so

# )

# # Odom tools

# rosbuild_add_executable(odom_saver
# src/tools/odom_saver.cpp
# )
# target_link_libraries(odom_saver
#    boost_system
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${OpenCV_LIBS} 

# )

# rosbuild_add_executable(odom_generator
# src/tools/odom_generator.cpp
# )
# target_link_libraries(odom_generator
#    boost_system
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${OpenCV_LIBS} 

# )

rosbuild_add_executable(tool_publish_images
   src/tools/tool_publish_images.cpp
)
target_link_libraries(tool_publish_images
   boost_system
   ${PROJECT_SOURCE_DIR}/../../../release/libutils.so
   ${OpenCV_LIBS} 
   ${cv_bridge_LIBRARIES}
   ${PROJECT_SOURCE_DIR}/../../../release/src/tum_rgbd/libtum_rgbd.so
)

rosbuild_add_executable(tool_save_bbox
   src/tools/tool_save_bbox.cpp
)
target_link_libraries(tool_save_bbox
   boost_system
   ${PROJECT_SOURCE_DIR}/../../../release/libutils.so
   ${OpenCV_LIBS} 
   ${cv_bridge_LIBRARIES}

)

# rosbuild_add_executable(pub_odom_withTimeStamp
# src/tools_homeDataset/pub_odom_withTimeStamp.cpp
# )
# target_link_libraries(pub_odom_withTimeStamp
#    boost_system
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${OpenCV_LIBS} 
#    ${cv_bridge_LIBRARIES}

# )

# rosbuild_add_executable(pub_bbox_withTimeStamp
# src/tools_homeDataset/pub_bbox_withTimeStamp.cpp
# )
# target_link_libraries(pub_bbox_withTimeStamp
#    boost_system
#    ${PROJECT_SOURCE_DIR}/../../lib/libutils.so
#    ${OpenCV_LIBS} 
#    ${cv_bridge_LIBRARIES}

# )

rosbuild_add_executable(debug_show_images
   src/tools/debug_show_images.cpp
)
target_link_libraries(debug_show_images
   ${OpenCV_LIBS} 
   ${cv_bridge_LIBRARIES}
)
