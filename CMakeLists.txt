cmake_minimum_required (VERSION 2.8) 
project (marcvox) 

set (CMAKE_CXX_LIBRARY_ARCHITECTURE x64)
set (CMAKE_INSTALL_PREFIX "./")

IF (WIN32)
set (CMAKE_PREFIX_PATH   "p:/opencv-2.4.8_64/"
	                 "p:/download/opencv-2.4.8/opencv/sources/"
                         "p:/OpenSceneGraph-3.3.0"
                         "p:/OpenSceneGraph-3.3.0/CMakeModules"
                         "p:/qt5stable/qtbase/lib/cmake"
                         "p:/tbb42_20140122oss")
# a little help to findtbb.cmake ...
find_library(TBB_LIBRARY_DEBUG tbb_debug HINTS p:/tbb42_20140122oss/lib/intel64/vc12)
find_library(TBB_LIBRARY tbb HINTS p:/tbb42_20140122oss/lib/intel64/vc12)
find_library(TBB_MALLOC_LIBRARY_DEBUG tbbmalloc_debug HINTS p:/tbb42_20140122oss/lib/intel64/vc12)
find_library(TBB_MALLOC_LIBRARY tbbmalloc HINTS p:/tbb42_20140122oss/lib/intel64/vc12)
set (TBB_INCLUDE_DIR      "p:/tbb42_20140122oss/include")
ENDIF()

IF (UNIX)
set (CMAKE_PREFIX_PATH "~/github/intel-tbb/")
set (TBB_PREFIX_RELEASE "~/github/intel-tbb/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.19.0_release/")
set (TBB_PREFIX_DEBUG "~/github/intel-tbb/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.19.0_debug/")
find_library(TBB_LIBRARY_DEBUG tbb_debug HINTS ${TBB_PREFIX_DEBUG})
find_library(TBB_LIBRARY tbb HINTS ${TBB_PREFIX_RELEASE})
find_library(TBB_MALLOC_LIBRARY_DEBUG tbbmalloc_debug HINTS ${TBB_PREFIX_DEBUG})
find_library(TBB_MALLOC_LIBRARY tbbmalloc HINTS ${TBB_PREFIX_RELEASE})
set (TBB_INCLUDE_DIR      ${CMAKE_PREFIX_PATH}/include)
ENDIF()
						  

FINd_PACKAGE(OpenCV)

set (TBB_FOUND Yes)
set (TBB_LIBRARIES        "${TBB_LIBRARY}"       "${TBB_MALLOC_LIBRARY}")
set (TBB_DEBUG_LIBRARIES  "${TBB_LIBRARY_DEBUG}" "${TBB_MALLOC_LIBRARY_DEBUG}" )

message (STATUS "TBB_FOUND ....................: ${TBB_FOUND}")
message (STATUS "TBB_LIBRARY_DEBUG ............: ${TBB_LIBRARY_DEBUG}")
message (STATUS "TBB_MALLOC_LIBRARY_DEBUG......: ${TBB_MALLOC_LIBRARY_DEBUG}")
message (STATUS "TBB_LIBRARY ..................: ${TBB_LIBRARY}")
message (STATUS "TBB_MALLOC_LIBRARY............: ${TBB_MALLOC_LIBRARY}")
message (STATUS "TBB_INCLUDE_DIR ..............: ${TBB_INCLUDE_DIR}")
message (STATUS "tbb_libraries.................: ${TBB_LIBRARIES}")
message (STATUS "tbb_debug_libraries...........: ${TBB_DEBUG_LIBRARIES}")
message (STATUS "TBB_INSTALL_DIR...............: ${TBB_INSTALL_DIR}")
message( STATUS "opencv_include_dirs...........: ${OpenCV_INCLUDE_DIRS}")
message( STATUS "opencv_libs...................: ${OpenCV_LIBS}")
message( STATUS "OpenCV_SHARED.................: ${OpenCV_SHARED}")
message( STATUS "CMAKE_INSTALL_PREFIX..........: ${CMAKE_INSTALL_PREFIX}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "p:/")

set (TASKS_FILES templeRing.tsk)
set_source_files_properties(templeRing.tsk PROPERTIES HEADER_FILE_ONLY TRUE)


SET(PROJECT_SOURCES  src/main.cpp 
					 src/imgprocess.h
					 src/ply.cpp
                     src/imgprocess.cpp
					 src/tasks.h
					 src/tasks.cpp)

ADD_EXECUTABLE(${PROJECT_NAME}  ${PROJECT_SOURCES} ${TASKS_FILES})  

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC "C:/Users/marc/Google Drive/My Documents/shared/projects/tclap-1.2.1/include")

target_link_libraries(${PROJECT_NAME} optimized ${TBB_LIBRARIES})
target_link_libraries(${PROJECT_NAME} debug ${TBB_DEBUG_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIR})
include_directories("P:/eigen-3.2.4")

install(TARGETS ${PROJECT_NAME}  DESTINATION bin/release CONFIGURATIONS Release)
install(FILES "p:/opencv-2.4.8_64/bin/Release/opencv_imgproc248.dll" DESTINATION bin/release CONFIGURATIONS Release)
install(FILES "p:/opencv-2.4.8_64/bin/Release/opencv_core248.dll" DESTINATION bin/release CONFIGURATIONS Release)
install(FILES "p:/opencv-2.4.8_64/bin/Release/opencv_highgui248.dll" DESTINATION bin/release CONFIGURATIONS Release)
install(FILES "P:/tbb42_20140122oss/bin/intel64/vc12/tbb.dll"  DESTINATION bin/release CONFIGURATIONS Release)
install(FILES "P:/tbb42_20140122oss/bin/intel64/vc12/tbbmalloc.dll"  DESTINATION bin/release CONFIGURATIONS Release)


