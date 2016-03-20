// +++++++++++++++++++++++
//  Marc Racicot  @ 2015
// +++++++++++++++++++++++
#pragma once

#define _USE_MATH_DEFINES

#include <memory>
#include <limits.h>
#include <vector>
#include <array>
#include <string>
#include <list>
#include <forward_list>
#include <map>
#include <valarray>
#include <functional>
#include <Eigen/Dense>  
#include <algorithm>    // std::sort
#include <math.h>
#include <atomic>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/atomic.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tasks.h"

const int MAX_LEVEL = 10;  //  this is the maximum level for the voxel octree
const float DEFAULT_COST = 1000.0f;
//
// Octree is composed of unsigned char, a bit field:
// bit 
//  0 : empty=0,      full=1
//  1 : notpartial=0, partial=1
//  2 :
//  3 :
//  4-7: viewpoint id that carve this voxel, will be used to get color of voxel
//
const unsigned char VOXEL_EMPTY   = 0x00;
const unsigned char VOXEL_FULL    = 0x01; // bit 0, empty
const unsigned char VOXEL_PARTIAL = 0x03; // bit 1, partial

#define ISVOXELEMPTY(OCENTRY) (((OCENTRY) & 0x01) == 0)
#define ISVOXELFULL(OCENTRY) (((OCENTRY) & 0x01) == 0x01)
#define ISVOXELPARTIAL(OCENTRY) (((OCENTRY) & 0x02) == 0x02)
#define VOXELEMPTY(VPID)  ((((VPID) & 0x0f)  << 4) | VOXEL_EMPTY)
#define VOXELVPID(OCENTRY)   ((OCENTRY) >> 4)
#define ADDPLYPOS(VPID,LVL,X,Y,Z) static_cast<int>( std::distance(listofpos.begin(), listofpos.push_back(plyPosition_s{VPID,LVL,X,Y,Z}) ) )
#define ADDPLYWORLDPOS(X,Y,Z) static_cast<int>( std::distance(listofpos.begin(), listofpos.push_back(plyPosition_s{-1,-1,X,Y,Z}) ) )
#define OFFST(DX,DY,DZ,VSL1,VSL2) ( ((DX)*((VSL1)*(VSL2))) + ((DY)*(VSL2)) + (DZ) )

struct plyPosition_s 
{ 
    int vpid;  
    int level; 
    double x; 
    double y; 
    double z; 
};

struct plyTriangle_s
{ 
    int t1; 
    int t2; 
    int t3; 
};

struct voxwork_s
{
    int split_level;
    int xsl;  
    int ysl;  
    int max_level;
    int oversize;
    int viewPointNdx;
};

using voxWork_t = voxwork_s;
using Matrix3_4d = Eigen::Matrix<double, 3, 4>;
using vectorOfMatrix4d_t = std::vector<Eigen::Matrix4d>;
using vecOfStrings_t = std::vector<std::string>;
using cameraSet_t = std::vector < std::tuple<int, double> >;
using dirMask_t = std::valarray < unsigned char >;
using voxWorkArr_t = std::vector < voxWork_t >;
using octree_t = std::vector < unsigned char > ;

struct cost_s
{
    int side;
    int depth;
    std::valarray < double > arr;
    voxWork_t voxwork;
};

struct surface_s
{
    int side;
    int depth;
    std::valarray<int> arr;
    voxWork_t voxwork;
};

struct camera_s
{
    int imgnb;                 // image number  (0..)
    std::string shortfilename; // image filename (path NOT included)
    std::string filename;      // image filename (path included)
    cv::Mat image;             // the original image
    Eigen::Matrix3d K;         // internal camera matrix 
    Eigen::Matrix3d R;         // Rotation matrix
    Eigen::Vector3d T;         // Translation vector
    Matrix3_4d P;              // 3x4 projection matrix 
    Eigen::Matrix4d Pinv;      // Inverse of Camera Matrix
    Eigen::Vector3d Origin;    // Camera Origin position
    Eigen::Vector3d c;         // camera vector c;
};

struct viewPoint_s
{
    cameraSet_t cameraset;    // vector of camera that are seeing this viewpoint
    double theta;             // angle of viewpoint
    vectorOfMatrix4d_t A;     // the A matrix, to convert viewpint coordinate into world coordinate)
    vectorOfMatrix4d_t Ainv;  // A inverse
    vectorOfMatrix4d_t B;     // the B matrix, to convert from a viewpoint coordinate into viewpoint-theta=0 coordinate
    Eigen::Vector3d v;        // virtual camera vector
    int id;                   // index in array
};

typedef struct
{
    int dA;       // delta y
    int dB;       // delta x
    int Asta;     // A start
    int Aend;     // A end     
    int Ainc;     // A increment
    int Bsta;     // B start 
    int Bend;     // B end
    int Binc;     // B increment
    int outer;    // outer loop (lut indexes for loop)
    int inner;    // inner loop
} dirinfo_t;


using plyPosition_t      = tbb::concurrent_vector<plyPosition_s>;
using plyTriangles_t     = tbb::concurrent_vector<plyTriangle_s>;
using plycallback        = std::function < void(int level, plyPosition_t& listofworldpos, plyTriangles_t & triangles) >;
using cost_t             = cost_s;
using surface_t          = surface_s;
using camera_t           = camera_s;
using cameraArr_t        = std::vector < std::unique_ptr<camera_t> > ;
using ViewPoint_t        = std::unique_ptr < viewPoint_s >;
using ViewPointArr_t     = std::vector < ViewPoint_t >;


// -----------------------------------------------------
// classes
// -----------------------------------------------------

class Images
{
public:
    unsigned int nbimages{ 0 };        // nb images, first parameter of txt file
    int          m_imgwidth{ 0 };      // width of the images
    int          m_imgheight{ 0 };     // height of the images
    cameraArr_t  m_imgsvec;            // image and camera information vector
    float        m_threshold{ 0.01f};  // this is the treshold value deciding if pixel is background or foreground.  The mask range is 0..1

    Eigen::Vector2d getScreenPosition(const Eigen::Vector4d &pw, int imgnum) const
    {
        Eigen::Vector3d psp = m_imgsvec[imgnum]->P * pw; // Screen perspective position
        Eigen::Vector2d pse = psp.head(2) / psp(2);      // Perspective to Euclidien
        return std::move(pse);
    }
};

class BoundingBox
{
public:
    std::array<std::array<double, 2>, 3> m_boundbox; // bounding box coordinate
    std::array<double, 3> m_boundboxsize;            // size of the box (delta x,...)
    double m_s{ 0.0 };                               // size of the cuboid for the bounding box.
    Eigen::Vector3d m_c;                             // bounding box center position
};

class Execution
{
public:
    virtual void initExecuteTask()=0;  // this populate the map 'm_cmdMap' with the string to lambda relations

protected:
    std::string m_directory; // to store results in a sub directory, this is the directory name
    std::map<std::string, std::function<bool(const vecOfStrings_t &)>> m_cmdMap;
};

class Octree 
{
public:
    Octree::Octree() {
        unsigned long startpos = 0;
        for (int i = 0; i <= MAX_LEVEL; i++)
        {
            voxelLevelStartPosition[i] = startpos;
            voxelSideAtLevel[i] = static_cast<unsigned long>(std::pow(2.0, static_cast<double>(i)));
            startpos += voxelSideAtLevel[i] * voxelSideAtLevel[i] * voxelSideAtLevel[i];
        }
    }
    octree_t m_octree; // the octree!
    int m_maxlevel{ 0 };     // this is the maximum level that needs to be reached to get voxel/pixl 1/1 diff
    std::array<int, (MAX_LEVEL + 1)> voxelSideAtLevel;
    std::array<int, (MAX_LEVEL + 1)> voxelLevelStartPosition;

    // lvl voxl   side        Nb Voxel      MB
    //  0	1	   1	        1	        1  
    //  1	2	   8  	        9	        1
    //  2	4	   64	        73	        1
    //  3	8	   512	        585	        1
    //  4	16	   4096	        4681	    1
    //  5	32	   32768	    37449	    1
    //  6	64	   262144	    299593	    1
    //  7	128	   2097152	    2396745	    3
    //  8	256	   16777216	    19173961	19
    //  9	512	   134217728	153391689	147
    //  10	1024   1073741824	1227133513	1171
    unsigned long getOctreSize(int level) {
        if (level <= 0) return 1;
        unsigned long s = voxelSideAtLevel[level]; 
        return s*s*s + getOctreSize(level - 1);
    }

    // returns the index in the octree storage where the voxel is store
    int voxelndx(int level, int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0) return -1;
        int voxside = voxelSideAtLevel[level];
        if (x >= voxside || y >= voxside || z >= voxside) return -1;
        return voxelLevelStartPosition[level] + x + y*voxside + z*voxside*voxside;
    }

    void percolate();
};

class ViewPoints : virtual public Octree {
public:
    ViewPointArr_t m_viewpointsvec; // viewpoints list

    void carve(const ViewPoint_t& vp, const surface_t& surface)
    {
        //----+--<xdl>---+----------+          /
        //    |XXXXXXXXXX|OO        |         /
        //  ^ |X (3,0) XX|OO        |        /
        // ydl|XXXXXXXXXX|OO        |    depth
        //  V |XXXXXXXXXX|OO        |      /
        //----+----------+----------+     /
        //    |OOOOOOOOOO|OO        |    /
        //    |OOOOOOOOOO|OO        |   /
        //    |          |          |  /
        //    |          |          | /
        //----+----------+----------+/
        //    |<--side---->|
        //    |<--vsdl-->|
        int vsdl = voxelSideAtLevel[surface.voxwork.max_level - surface.voxwork.split_level];
        int vsl_maxlvl = surface.depth;
        int basePosX = vsdl * surface.voxwork.xsl;
        int basePosY = vsdl * surface.voxwork.ysl;
        for (int xdl = 0; xdl < surface.side; xdl++)
        {
            int xml{ basePosX + xdl };
            for (int ydl = 0; ydl < surface.side; ydl++)
            {
                int yml { basePosY + ydl };
                int dirmaskndx = (surface.side * xdl) + ydl;
                for (int zml = 0; zml < surface.arr[dirmaskndx]; zml++)
                {
                    int voxndx = voxelndxVP(m_maxlevel, vp, xml, yml, zml);
                    m_octree[voxndx] = VOXELEMPTY(vp->id);;
                }
            }
        }
    }

    template<typename T>
    Eigen::Vector4d getWorldPosition(int level, const ViewPoint_t&vp, T xvp, T yvp, T zvp) const
    {
        Eigen::Matrix4d& A = vp->A[level];
        Eigen::Vector4d pw = A * Eigen::Vector4d{ static_cast<double>(xvp), static_cast<double>(yvp), static_cast<double>(zvp), 1.0 };
        return std::move(pw);
    }

    std::tuple<int, int, int> getVp0Position(int level, const ViewPoint_t&vp, int xvp, int yvp, int zvp) const
    {
        Eigen::Vector4d pvp0 = vp->B[level] * Eigen::Vector4d{ static_cast<double>(xvp), static_cast<double>(yvp), static_cast<double>(zvp), 1.0 };
        // Ok, when doing static_cast<int>(pvp0[0]), it will floor... So a value like:
        // 510.99999999999994 will turn out to be 510 which is not what we want.
        // to work around this, let's std::floor(pvp + 0.5) before static_cast
        double px = std::floor(pvp0[0] + 0.5);
        double py = std::floor(pvp0[1] + 0.5);
        double pz = std::floor(pvp0[2] + 0.5);
        return std::move(std::make_tuple(static_cast<int>(px), static_cast<int>(py), static_cast<int>(pz)));
    }

    // performs a viewpoint conversion (from vp#n to vp#0) before calling voxelndx
    int voxelndxVP(int level, const ViewPoint_t &vp, int xvp, int yvp, int zvp) const
    {
        Eigen::Vector4d pvp0 = vp->B[level] * Eigen::Vector4d{ static_cast<double>(xvp), static_cast<double>(yvp), static_cast<double>(zvp), 1.0 };
        return voxelndx(level, static_cast<int>(pvp0[0]), static_cast<int>(pvp0[1]), static_cast<int>(pvp0[2]));
    }

    // let's find the voxels at level 'split_level' that are not empty and not full. 
    voxWorkArr_t searchNotEmptyVoxels(int split_level, const ViewPoint_t&vp, int maxlevel, int oversize, int viewPointNdx)
    {
        int vsl_splitlvl = voxelSideAtLevel[split_level];
        voxWorkArr_t voxWorkArr;
        for (int xsl = 0; xsl < vsl_splitlvl; xsl++)
            for (int ysl = 0; ysl < vsl_splitlvl; ysl++)
            {
                for (int zsl = 0; zsl < vsl_splitlvl; zsl++) {
                    int voxndx = voxelndxVP(split_level, vp, xsl, ysl, zsl);
                    if (voxndx == -1) continue;
                    if ( ! ISVOXELEMPTY(m_octree[voxndx]))
                    {
                        voxWorkArr.push_back({ split_level, xsl, ysl, maxlevel, oversize, viewPointNdx });
                        break;
                    }
                }
            }
        return std::move(voxWorkArr);
    }
};


class Costs_c :
    virtual public Octree,     // for m_octree, ...
    virtual public ViewPoints, // for getVp0Position
    virtual private Images
{
public:
    // ------------------------------------------------------------------------------------------
    //      |xo              |x1
    // y0---+----------------+---
    //      |                |
    //      |                |
    //      |          x     |
    // y1---+----------------+---
    //      |                |
    //
    float bilinear(const Eigen::Vector2d& pse, const cv::Mat& img, float defaultval)
    {
        using T = float;
        T retf;
        double pfx = pse[0];
        double pfy = pse[1];
        int x = (int)floor(pfx);
        int y = (int)floor(pfy);
        if (x < 0 || y < 0) return defaultval;
        if (x >= img.cols || y >= img.rows) return defaultval;
        int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
        int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
        int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
        int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);
        float a = (float)pfx - (float)x;
        float c = (float)pfy - (float)y;
        retf = (img.at<T>(y0, x0) * (1.f - a) + img.at<T>(y0, x1) * a) * (1.f - c)
            + (img.at<T>(y1, x0) * (1.f - a) + img.at<T>(y1, x1) * a) * c;
        return retf;
    }

    bool bilinear(const Eigen::Vector2d& pse, const cv::Mat& img, float defaultval, float(&results)[4])
    {
        using T = float;
        double pfx = pse[0];
        double pfy = pse[1];
        int x = (int)floor(pfx);
        int y = (int)floor(pfy);

        results[0] = results[1] = results[2] = defaultval;
        if (x < 0 || y < 0) return false;
        if (x >= img.cols || y >= img.rows) return false;

        int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
        int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
        int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
        int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);
        float a = (float)pfx - (float)x;
        float c = (float)pfy - (float)y;

        auto o = std::numeric_limits<cv::Vec3b::value_type>::max();

        results[2] = (img.at<cv::Vec3b>(y0, x0)[0] * (1.0f - a) + img.at<cv::Vec3b>(y0, x1)[0] * a) * (1.0f - c)
            + (img.at<cv::Vec3b>(y1, x0)[0] * (1.0f - a) + img.at<cv::Vec3b>(y1, x1)[0] * a) * c;
        results[2] /= static_cast<float>(o);
        results[1] = (img.at<cv::Vec3b>(y0, x0)[1] * (1.0f - a) + img.at<cv::Vec3b>(y0, x1)[1] * a) * (1.0f - c)
            + (img.at<cv::Vec3b>(y1, x0)[1] * (1.0f - a) + img.at<cv::Vec3b>(y1, x1)[1] * a) * c;
        results[1] /= static_cast<float>(o);
        results[0] = (img.at<cv::Vec3b>(y0, x0)[2] * (1.0f - a) + img.at<cv::Vec3b>(y0, x1)[2] * a) * (1.0f - c)
            + (img.at<cv::Vec3b>(y1, x0)[2] * (1.0f - a) + img.at<cv::Vec3b>(y1, x1)[2] * a) * c;
        results[0] /= static_cast<float>(o);

        return true;
    }

};

class Ply :
    virtual public Octree,     // for m_octree, ...
    virtual public ViewPoints, // for getVp0Position
    virtual public Execution,   // for  m_directory
    virtual private Images,
    virtual public Costs_c
{
public:
    void savefilePly(const std::string& filename, int level, const ViewPoint_t& vp, bool StopAtFirstFace = false, const std::vector<std::tuple<int, int, int>> * vp0special = nullptr, int maxlevel = -1, bool allres = true, plycallback myfunc = nullptr);

    void savePlyCompleteAllLevel(const std::string & addedname) // save the ply for each level
    {
        tbb::parallel_for(0, (int)(m_maxlevel + 1), 1, [&](int level) 
        {
            std::string filename = "results/" + m_directory + "/" + addedname + std::to_string(level) + ".ply";
            savefilePly(filename, level, m_viewpointsvec[0], false);
        });
    }
};


class ImageProcessing :
    public imgproTasks,
    private BoundingBox,
    virtual private Images,
    virtual public Execution,
    virtual public Octree,
    virtual public ViewPoints,
    public Ply,
    virtual public Costs_c
{
public:
    ImageProcessing();
    ~ImageProcessing() {}

private:
    void initExecuteTask();  // this populate the map 'm_cmdMap' with the string to lambda relations
    bool executeTask(const vecOfStrings_t &vst); // this will execute the tasks listed in the taskfile 
};


