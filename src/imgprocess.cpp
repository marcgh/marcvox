// +++++++++++++++++++++++
//  Marc Racicot  @ 2015
// +++++++++++++++++++++++
#include "imgprocess.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <iterator>
#include <stdexcept>
#include <random>
#include <atomic>
#include <mutex>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <direct.h>

#undef min // to prevent strange warning ....
#undef max

using std::istringstream;
using std::copy;
using std::stringstream;
using std::istream_iterator;
using std::back_inserter;
using std::ifstream;
using std::string;
using std::vector;
using std::pair;
using std::cout;
using std::endl;
using std::list;
using cv::Mat;

using namespace tbb;
using namespace std::chrono;

ImageProcessing::ImageProcessing()
{
    std::cout << "Initializing tasks map\n";
    initExecuteTask();
}

bool ImageProcessing::executeTask(const vecOfStrings_t &vst)
{
    if (vst.size() == 0 || vst[0].size() == 0)
        return true; //empty line
    std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::high_resolution_clock::now();
    std::stringstream buffer;
    std::cout << "Executing \"" << vst[0] << "\"";
    bool result=false;
    try
    {
        result = m_cmdMap[vst[0]](vst);

        std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::high_resolution_clock::now();
        float fduration1 = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        fduration1 /= 1000.0;
        std::cout << ": " << fduration1 << " sec";
    }
    catch (const std::bad_function_call& e)
    {
       std::cout << ": Error " << e.what() << " ";
    }
    catch (...)
    {
        return false;
    }
    std::cout << "\n";
    return true;
}

void loadImage(
    const vecOfStrings_t & imagesInfoVector,
    std::unique_ptr<camera_t>& pimginfo)
{
    // 
    //  order in vector: "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
    //  The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.
    //  images have been corrected to remove radial distortion.
    //
    pimginfo->image = cv::imread(pimginfo->filename.c_str());
    if (!pimginfo->image.data)
    {
        throw "Could not open or find the image";
    }

    vector<vector<int>> pos{ { 0, 0, 2 }, { 0, 1, 3 }, { 0, 2, 4 }, { 1, 0, 5 }, { 1, 1, 6 }, { 1, 2, 7 }, { 2, 0, 8 }, { 2, 1, 9 }, { 2, 2, 10 } };
    for (const vector<int> & it : pos)
    {
        // Internal Parameter Camera Matrix K
        // +-           -+
        // |  fx   0  Cx |
        // |   0  fy  Cy |
        // |   0   0   1 |
        // +-           -+
        pimginfo->K(it[0], it[1]) = atof(imagesInfoVector.at(pimginfo->imgnb * 22 + it[2]).c_str());
        
        // Rotation Camera Matrix R (external parameter)
        // +-           -+
        // | r11 r12 r13 |
        // | r21 r22 r23 |
        // | r31 r32 r33 |
        // +-           -+
        pimginfo->R(it[0], it[1]) = atof(imagesInfoVector.at(pimginfo->imgnb * 22 + it[2] + 9).c_str());
    }

    // Translation Camera Matrix T (external parameter)
    // +-  -+
    // | t1 |
    // | t2 |
    // | t3 |
    // +-  -+
    vector<vector<int>> posT{ { 0, 0, 20 }, { 1, 0, 21 }, { 2, 0, 22 } };
    for (const auto & it : posT)
        pimginfo->T(it[0]) = atof(imagesInfoVector.at(pimginfo->imgnb * 22 + it[2]).c_str());

    // Matrix E (for external,   R | Rt)
    // +-              -+
    // | r11 r12 r13 t1 |
    // | r21 r22 r23 t2 |
    // | r31 r32 r33 t3 |
    // +-              -+
    Matrix3_4d E;
    E << pimginfo->R, pimginfo->T;

    // [3x3] * [3x4]
    pimginfo->P = pimginfo->K * E;
    
    Eigen::Matrix4d tmp;
    tmp.block<3,4>(0,0) = pimginfo->P;
    tmp.row(3) << 0.0, 0.0, 0.0, 1.0;
    pimginfo->Pinv = tmp.inverse();

    // On inverse(P), the last column is the origin of the camera
    pimginfo->Origin = pimginfo->Pinv.col(3).head(3);
    
    double f = 1.0;
    double xmiddle = f*static_cast<double>(pimginfo->image.cols) / 2.0;
    double ymiddle = f*static_cast<double>(pimginfo->image.rows) / 2.0;
    
    Eigen::Vector4d center(xmiddle, ymiddle, f, 1.0);
    Eigen::Vector4d centerw = pimginfo->Pinv * center;
    pimginfo->c = centerw.head(3) - pimginfo->Origin;
    pimginfo->c.normalize(); // camera vector (point from camera's origin toward camera sensor center point)
}

void Octree::percolate()  // Octree percolation
{
    for (int level = m_maxlevel - 1; level >= 0; --level)
    {
        int nxtlevel = level + 1;
        int voxside = voxelSideAtLevel[level];
        tbb::parallel_for(blocked_range<int>(0, voxside), [&](const blocked_range<int>& r)
        {
            int offsets[8];
            for (int x = r.begin(); x != r.end(); ++x)
                for (int y = 0; y < voxside; y++)
                    for (int z = 0; z < voxside; z++)
                    {
                        offsets[0] = voxelndx(level + 1, 2 * x, 2 * y, 2 * z);
                        offsets[1] = voxelndx(level + 1, 2 * x + 1, 2 * y, 2 * z);
                        offsets[2] = voxelndx(level + 1, 2 * x, 2 * y + 1, 2 * z);
                        offsets[3] = voxelndx(level + 1, 2 * x + 1, 2 * y + 1, 2 * z);
                        offsets[4] = voxelndx(level + 1, 2 * x, 2 * y, 2 * z + 1);
                        offsets[5] = voxelndx(level + 1, 2 * x + 1, 2 * y, 2 * z + 1);
                        offsets[6] = voxelndx(level + 1, 2 * x, 2 * y + 1, 2 * z + 1);
                        offsets[7] = voxelndx(level + 1, 2 * x + 1, 2 * y + 1, 2 * z + 1);
                        int nb_full = 0;
                        int nb_empty = 0;
                        for (auto r : offsets)
                        {
                            if (ISVOXELEMPTY(m_octree[r])) nb_empty++;
                            if (ISVOXELFULL(m_octree[r]))  nb_full++;
                        }
                        int voxoff = voxelndx(level, x, y, z);
                        if (nb_empty == 8)     m_octree[voxoff] = VOXEL_EMPTY;
                        else if (nb_full == 8) m_octree[voxoff] = VOXEL_FULL;
                        else                   m_octree[voxoff] = VOXEL_PARTIAL;
                    }
        });
    }
}


void ImageProcessing::initExecuteTask()
{
    using vecofstrings = const std::vector < std::string > &;

    // ----------------------------------------------------------------------------------------
    // initialize model with specified bounding box
    m_cmdMap["boundbox"] = [this](vecofstrings vst)->bool
    {
        double x1, x2, y1, y2, z1, z2;
        istringstream(vst[1]) >> x1;
        istringstream(vst[2]) >> x2;
        istringstream(vst[3]) >> y1;
        istringstream(vst[4]) >> y2;
        istringstream(vst[5]) >> z1;
        istringstream(vst[6]) >> z2;

        m_boundbox = { { { { x1, x2 } }, { { y1, y2 } }, { { z1, z2 } } } };
        double dx = abs(m_boundbox[0][1] - m_boundbox[0][0]); // delta x
        double dy = abs(m_boundbox[1][1] - m_boundbox[1][0]); // delta y
        double dz = abs(m_boundbox[2][1] - m_boundbox[2][0]); // delta z
        m_boundboxsize = { { dx, dy, dz } }; 

        // the specified bouding box is not squared, we need a square bounding box to work with cuboid voxels.
        m_s = std::max(std::max(dx, dy), dz);
        m_c << std::min(m_boundbox[0][1], m_boundbox[0][0]) + (dx / 2.0),
               std::min(m_boundbox[1][1], m_boundbox[1][0]) + (dy / 2.0),
               std::min(m_boundbox[2][1], m_boundbox[2][0]) + (dz / 2.0);
        std::cout << "Bouding box center : " << m_c << "\n";
        std::cout << "Bouding box size : " << m_s << "\n";
        return true;
    };

    // ----------------------------------------------------------------------------------------
    // create a directory where the results (images, ply, etc..) will be saved
    m_cmdMap["directory"] = [this](vecofstrings vst)->bool
    {
        if (vst.size() > 1)
            m_directory = vst[1];
        else
        {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::stringstream ss;
            ss << std::put_time(&tm, "%Y-%j_%H-%M-%S");
            m_directory = ss.str();
        }
        std::string pathname = "results/" + m_directory + "/";
        int res = _mkdir(pathname.c_str());
        if (res == 0)
            return true;
        return false;
    };

    // ----------------------------------------------------------------------------------------
    // process image file containing camera matrices and image filename
    m_cmdMap["setimagefile"] = [this](vecofstrings vst)->bool
    {
        vecOfStrings_t imagesInfoVector;
        try { 
            //
            // 312
            // temple0001.png 1520.400000 0.000000 302.320000 0.000000 1525.900000 246.870000 0.000000 0.000000 1.000000 0.01551372092999463200 0.99884343581246959000 - 0.04550950666890610900 0.99922238739871228000 - 0.01713749902859566800 - 0.03550952897832390700 - 0.03624837905512174500 - 0.04492323298011671700 - 0.99833258894743582000 - 0.05998547900141842900 0.00400788029504099870 0.57088647431543438000
            // temple0002.png 1520.400000 0.000000 302.320000 0.000000 1525.900000 246.870000 0.000000 0.000000 1.000000 0.01614490437974924100 0.99884677638989772000 - 0.04521569813768747100 0.99856922398083869000 - 0.01380176413826810800 0.05166252244109931900 0.05097888759941932700 - 0.04598509108593063600 - 0.99764047853770654000 - 0.05998004112555456500 0.00374555199382083440 0.57175508950314680000
            // temple0003.png 1520.400000 0.000000 302.320000 0.000000 1525.900000 246.870000 0.000000 0.000000 1.000000 0.01705593519923021000 0.99885654318139749000 - 0.04466207807736556600 0.98284888271077153000 - 0.00854576872467372430 0.18421466714431800000 0.18362235383707279000 - 0.04703802696344126100 - 0.98187079353177587000 - 0.05997257241836466400 0.00319966369542803480 0.57303277427659483000
            // temple0004.png 1520.400000 0.000000 ...
            // ...
            //
            std::ifstream inFile(vst[1].c_str());
            std::string path(vst[1].c_str());
            std::string imagesPath = path.substr(0, path.find_last_of("\\/"));

            // copy all items into a vector
            copy(istream_iterator<string>(inFile), istream_iterator<string>(), back_inserter(imagesInfoVector));
            stringstream convert(imagesInfoVector.at(0));
            if (!(convert >> nbimages)) //give the value to Result using the characters in the string
                nbimages = 0;//if that fails set Result to 0
            m_imgsvec.resize(nbimages);
            for (unsigned i = 0; i < nbimages; i++)
            {
                std::unique_ptr<camera_t> iinfo{ new camera_t };
                iinfo->imgnb = i;
                m_imgsvec[i] = std::move(iinfo);
                m_imgsvec[i]->shortfilename = imagesInfoVector.at(i * 22 + 1);
                m_imgsvec[i]->filename = imagesPath + "/" + imagesInfoVector.at(i * 22 + 1);
            }

            tbb::parallel_for(0, (int)nbimages, 1, [this, &imagesInfoVector](int i)
            //for (unsigned int i = 0; i < nbimages; i++)
            {
                loadImage(imagesInfoVector,m_imgsvec[i]);
            } );
            m_imgwidth = m_imgsvec[0]->image.cols;
            m_imgheight = m_imgsvec[0]->image.rows;
        }
        catch (...) {
            return false;
        }
        return true;
    };

    // ----------------------------------------------------------------------------------------
    // Create the virtual camera or viewpoints.  They are specified by their theta angles
    // call example: theta("0", "1.57079632679", "3.14159265359", "4.71238898038");
    m_cmdMap["theta"] = [this](vecofstrings vst)->bool
    {
        if (vst.size() > 1)
        {
            for (int i = 1; i < vst.size(); i++)
            {
                ViewPoint_t vp{ new viewPoint_s };
                vp->theta = atof(vst[i].c_str());
                vp->id = i - 1;
                std::cout << "Virtual camera : " << vp->theta << "\n";

                auto getMatA = [](int level, double s, double theta, const Eigen::Vector3d& center)->Eigen::Matrix4d
                {
                    Eigen::Matrix4d A;
                    double twotol = std::pow(2.0, level);
                    double costheta = std::cos(theta);
                    double sintheta = std::sin(theta);
                    double pm = std::pow(2.0, -level - 1) * (twotol - 1) * s;
                    A = Eigen::Matrix4d::Constant(0.0);
                    A(0, 0) = (s * costheta) / twotol;
                    A(0, 2) = -(s * sintheta) / twotol;
                    A(0, 3) = center(0) - pm*costheta + pm*sintheta;
                    A(1, 1) = -(s) / twotol;
                    A(1, 3) = center(1) + pm;
                    A(2, 0) = A(0, 2);
                    A(2, 2) = -A(0, 0);
                    A(2, 3) = center(2) + pm*costheta + pm*sintheta;
                    A(3, 3) = 1.0;
                    return std::move(A);
                };
                auto getMatB = [](int level, double s, double theta)->Eigen::Matrix4d
                {
                    Eigen::Matrix4d B;
                    double twotol = std::pow(2.0, level);
                    double costheta = std::cos(theta);
                    double sintheta = std::sin(theta);
                    B = Eigen::Matrix4d::Constant(0.0);
                    B(0, 0) = costheta;
                    B(0, 2) = -sintheta;
                    B(0, 3) = -0.5*(twotol - 1.)*(costheta - sintheta - 1.);
                    B(1, 1) = 1.0;
                    B(2, 0) = sintheta;
                    B(2, 2) = costheta;
                    B(2, 3) = -0.5*(twotol - 1.)*(costheta + sintheta - 1.);
                    B(3, 3) = 1.0;
                    return std::move(B);
                };

                int level = 0;
                Eigen::Matrix4d A = getMatA(level, m_s, vp->theta, m_c);
                Eigen::Matrix4d B = getMatB(level, m_s, vp->theta);
                Eigen::Matrix4d Ainv = A.inverse();
                // the matrix A gives the center position of a voxel.  Taking the voxel(0,0) at level 0 will
                // give the center of the bounding box.

                Eigen::Vector4d head = A * Eigen::Vector4d(0, 0, 0, 1);
                Eigen::Vector4d tail = A * Eigen::Vector4d(0, 0, 0.4, 1);
                vp->v = (head - tail).head(3);
                vp->v.normalize();
                //std::cout << "v vector:" << vp->v << "\n" << A << std::endl;
                vp->A.push_back(std::move(A));
                vp->Ainv.push_back(std::move(Ainv));
                vp->B.push_back(std::move(B));

                // calculate the cameras that can be used for this viewpoint
                for (int i = 0; i < m_imgsvec.size(); i++)
                {
                    //
                    // Cos(eta) will provide and angle between pi/2 and pi.  The actual angle is pi-acos(eta).
                    // dividing  by the maximum range (pi/2) will give a vale from 0..1 where 1.0 is a good choice and 0 is a poor choice.
                    double coseta = vp->v.dot(m_imgsvec[i]->c);
                    double value = 1.0 - ((M_PI - std::acos(coseta)) / M_PI_2);
                    if (coseta < 0.0) {
                        vp->cameraset.push_back(std::tuple<int, double>(i, value));// keep camera id and value
                    }
                }

                // sort cameras by their values
                std::sort(vp->cameraset.begin(), vp->cameraset.end(), [](std::tuple<int, double> a, std::tuple<int, double> b) { return std::get<1>(b) < std::get<1>(a); });
                for (auto ig : vp->cameraset) {
                    std::cout << "     camera " << m_imgsvec[std::get<0>(ig)]->shortfilename.c_str() << " value " << std::get<1>(ig) << "\n";
                }

                //
                // Now that we know the camera that can be used for this viewpoint, let's take the best one (highest value)
                // and calculate the number of level we have to go to reach a delta of 1 pixel for a delta of one voxel.
                //
                // calculate A matrices for all 10 level and keep that.  But make a note of the real
                // maximum needed (it should be less than 10).
                //
                bool foundlevel = false;
                for (level = 1; level <= MAX_LEVEL; level++)
                {
                    double voxposmiddle = std::pow(2.0, level) / 2.0;
                    A = getMatA(level, m_s, vp->theta, m_c);  // 4x4
                    // world position
                    Eigen::Vector4d pw1 = A * Eigen::Vector4d{ voxposmiddle, voxposmiddle, 0, 1 };
                    Eigen::Vector4d pw2 = A * Eigen::Vector4d{ voxposmiddle + 1, voxposmiddle, 0, 1 };
                    // Screen perspective position
                    Eigen::Vector3d ps1p = m_imgsvec[0]->P * pw1;
                    Eigen::Vector3d ps2p = m_imgsvec[0]->P * pw2;
                    // Perspective to Euclidien
                    Eigen::Vector2d ps1e = ps1p.head(2) / ps1p(2);;
                    Eigen::Vector2d ps2e = ps2p.head(2) / ps2p(2);;
                    // ditance between to voxels in pixels
                    double dist = (ps1e - ps2e).norm();
                    if (dist < 1.0 && !foundlevel)
                    {
                        foundlevel = true;
                        if (level > m_maxlevel)
                        {

                            m_maxlevel = level;  /* <  */
                            if (m_maxlevel >= MAX_LEVEL) m_maxlevel = MAX_LEVEL-1;
                            std::cout << " max_level adjusted at " << m_maxlevel << "\n";
                        }
                    }
                    std::cout << dist << "\n";
                    Ainv = A.inverse();
                    vp->A.push_back(std::move(A));
                    vp->Ainv.push_back(std::move(Ainv));

                    Eigen::Matrix4d B = getMatB(level, m_s, vp->theta);
                    vp->B.push_back(std::move(B));
                }
                m_viewpointsvec.push_back(std::move(vp));
            }

            // With the maximum level calculated, we can allocate the voxels (the mask).
            // There will be one byte per voxel per level
            unsigned long octreeSize = getOctreSize(m_maxlevel);
            m_octree.resize(octreeSize, VOXEL_FULL);
        }
        return true;
    };

    // ----------------------------------------------------------------------------------------
    m_cmdMap["convexhull"] = [this](vecofstrings vst)->bool
    {
        int morphosize = 12;
        if (vst.size() > 1)
        {
            istringstream(vst[1]) >> m_threshold; 
            std::cout << "Using threshold " << m_threshold << "\n";
        }
        if (vst.size() > 2)
        {
            istringstream(vst[2]) >> morphosize;
            std::cout << "Using morphosize " << morphosize << "\n";
        }

        auto toAlphaMask = [=](const cv::Vec3b & img, float mask, cv::Vec4b& bgra){
            if (mask < m_threshold)
                bgra[0] = bgra[1] = bgra[2] = 0;
            else
                bgra[0] = bgra[1] = bgra[2] = 255;
            bgra[3] = 0xFF;
        };
        auto toImgAlpha = [=](const cv::Vec3b & img, float mask, cv::Vec4b& bgra){
            bgra[0] = img[0]; bgra[1] = img[1]; bgra[2] = img[2];
            if (mask < m_threshold)
                bgra[3] = 0x00;
            else
                bgra[3] = 0xFF;
        };

        // create all mask for all camera 
        std::vector<cv::Mat> masks(nbimages);
        for (int imgnum = 0; imgnum < masks.size(); imgnum++)
        {
            if (masks[imgnum].data == nullptr)
            {
                unsigned short maxSum = 0;
                cv::Mat imgsum(m_imgheight, m_imgwidth, cv::DataType<float>::type);
                for (int row = 0; row < m_imgheight; row++)
                    for (int col = 0; col < m_imgwidth; col++)
                    {
                        const cv::Vec3b & pixel = m_imgsvec[imgnum]->image.at<cv::Vec3b>(row, col);
                        unsigned short sum = pixel[0] + pixel[1] + pixel[2];
                        maxSum = std::max(maxSum, sum);
                        imgsum.at<float>(row, col) = static_cast<float>(sum);
                    }
                // normalized
                for (int row = 0; row < m_imgheight; row++)
                    for (int col = 0; col < m_imgwidth; col++)
                        imgsum.at<float>(row, col) /= static_cast<float>(maxSum);

                // morphological to close the hole inside image
                cv::Mat element(morphosize, morphosize, CV_32F, cv::Scalar(2.0));
                cv::morphologyEx(imgsum, masks[imgnum], cv::MORPH_CLOSE, element);
            }
        }

        // iterate on each voxel, at max_level, and check if a mask reports it as background
        int voxside = voxelSideAtLevel[m_maxlevel];
        for (int imgnum = 0; imgnum < m_imgsvec.size(); imgnum++)
        {
            tbb::parallel_for(0, (int)voxside, 1, [this, &voxside, &imgnum, &masks](int x) 
            {
                for (int  y = 0; y < voxside; y++)
                    for (int z = 0; z < voxside; z++)
                    {
                        int vxlndx = voxelndx(m_maxlevel, x, y, z);
                        if (ISVOXELEMPTY(m_octree[vxlndx])) continue;
                        Eigen::Vector4d pw  = getWorldPosition(m_maxlevel, m_viewpointsvec[0], x, y, z);
                        Eigen::Vector2d pse = getScreenPosition(pw, imgnum);
                        float value = bilinear(pse, masks[imgnum], 0.f);
                        if (value < m_threshold) {
                            m_octree[vxlndx] = VOXEL_EMPTY;
                        }
                    }
            });
            std::cout << "[" << std::setprecision(1) << std::fixed  << static_cast<float>(imgnum + 1) / static_cast<float>(m_imgsvec.size()) * 100.0f << "%]";
        } 
        std::cout << "\n";

        percolate(); // Octree percolation

        savePlyCompleteAllLevel("ply_level_");
        return true;
    };


}

