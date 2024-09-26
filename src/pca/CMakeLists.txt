# set(CMAKE_BUILD_TYPE Debug)
FIND_PACKAGE( PCL 1.10 REQUIRED )


add_library(EllipsoidExtractor SHARED
    EllipsoidExtractor.cpp
    EllipsoidExtractorMultiPlanes.cpp
    EllipsoidExtractorEdges.cpp
    EllipsoidExtractorLocalOptimization.cpp
)

target_link_libraries(EllipsoidExtractor
    pybind11::embed
    ${OpenCV_LIBS}
    symmetry
    utils
)
