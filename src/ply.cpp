// +++++++++++++++++++++++
//  Marc Racicot  @ 2015
// +++++++++++++++++++++++
#include <string>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include "tasks.h"
#include "tbb/tbb.h"
#include "tbb/blocked_range.h"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_vector.h"
#include "imgprocess.h"
#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <iomanip> // put_time

using std::string;
using std::vector;

//ply
//format ascii 1.0           { ascii / binary, format version number }
//comment made by Greg Turk{ comments keyword specified, like all lines }
//comment this file is a cube
//element vertex 8           { define "vertex" element, 8 of them in file }
//property float x           { vertex contains float "x" coordinate }
//property float y           { y coordinate is also a vertex property }
//property float z           { z coordinate, too }
//element face 6             { there are 6 "face" elements in the file }
//property list uchar int vertex_index { "vertex_indices" is a list of ints }
//end_header{ delimits the end of the header }
//new header:
//ply
//format ascii 1.0
//comment Created with the Wolfram Language : www.wolfram.com
//element vertex 23984
//property float32 x
//property float32 y
//property float32 z
//property uint8 red
//property uint8 green
//property uint8 blue
//element face 47904
//property list uint8 uint32 vertex_indices
//end_header



void Ply::savefilePly(
    const std::string& filename,
    int level, 
    const ViewPoint_t& vp,
    bool StopAtFirstFace,
    const std::vector<std::tuple<int, int, int>> * vp0special,
    int maxlevel,
    bool allres,
    plycallback myfunc)
{
        plyPosition_t listofpos;
        plyTriangles_t triangles;
        
        auto processvoxel = [this, &listofpos, &triangles](int level, int x, int y, int z)->bool
        {
            //        
            //      +-------+-------+
            //     /| c5   /| c6   /|
            //    +-------+-------+ |
            //   /|      /|      /|-+           (c7 an c8 hidden behind)
            //  +-------+-------+------------>X
            //  | +-----|-+-----|-+ |
            //  |/| c1  |/| c2  |/| +
            //  +-------+-------+ |/
            //  | +-----|-+-----|-+
            //  |/  c3  |/  c4  |/
            //  +-------+-------+
            //  |
            //  V y
            //   
            // voxel coordinates are in the middle of the voxels
            int c1 = voxelndx(level, x, y, z);
            int c2p = voxelndx(level, x - 1, y, z);
            int c2 = voxelndx(level, x + 1, y, z);
            int c3p = voxelndx(level, x, y - 1, z);
            int c3 = voxelndx(level, x, y + 1, z);
            int c5p = voxelndx(level, x, y, z - 1);
            int c5 = voxelndx(level, x, y, z + 1);

            if (c1 != -1 && (!ISVOXELEMPTY(m_octree[c1]))/* != VOXEL_EMPTY*/)
            {
                // surface are created from 6 directions (sides of cube)
                double xd = static_cast<double>(x);
                double yd = static_cast<double>(y);
                double zd = static_cast<double>(z);
                if (c2p == -1 || ISVOXELEMPTY(m_octree[c2p])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c2p == -1) ? -1 : VOXELVPID(m_octree[c2p]);
                    int t1 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd - 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd + 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd - 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd + 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                if (c2 == -1 || ISVOXELEMPTY(m_octree[c2])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c2 == -1) ? -1 : VOXELVPID(m_octree[c2]);
                    int t1 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd + 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd - 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd + 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd - 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                if (c3p == -1 || ISVOXELEMPTY(m_octree[c3p])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c3p == -1) ? -1 : VOXELVPID(m_octree[c3p]);
                    int t1 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd - 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd + 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd - 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd + 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                if (c3 == -1 || ISVOXELEMPTY(m_octree[c3])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c3 == -1) ? -1 : VOXELVPID(m_octree[c3]);
                    int t1 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd + 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd - 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd + 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd - 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                if (c5p == -1 || ISVOXELEMPTY(m_octree[c5p])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c5p == -1) ? -1 : VOXELVPID(m_octree[c5p]);
                    int t1 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd - 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd - 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd - 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd - 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                if (c5 == -1 || ISVOXELEMPTY(m_octree[c5])/* == VOXEL_EMPTY*/)
                {
                    int vpid = (c5 == -1) ? -1 : VOXELVPID(m_octree[c5]);
                    int t1 = ADDPLYPOS(vpid, level, xd - 0.5, yd - 0.5, zd + 0.5);
                    int t2 = ADDPLYPOS(vpid, level, xd + 0.5, yd - 0.5, zd + 0.5);
                    int t3 = ADDPLYPOS(vpid, level, xd - 0.5, yd + 0.5, zd + 0.5);
                    int t4 = ADDPLYPOS(vpid, level, xd + 0.5, yd + 0.5, zd + 0.5);
                    triangles.push_back({ t1, t2, t4 });
                    triangles.push_back({ t4, t3, t1 });
                }
                return true;
            }
            return false;
        };
        int voxside = voxelSideAtLevel[level];

#ifdef SINGLE_THREAD_PLY
        for (int x = 0; x < voxside; ++x)
#else
        tbb::parallel_for(0, (int)voxside, 1, [&](int x)
#endif
        {
            for (int y = 0; y < voxside; ++y)
            {
                for (int z = 0; z < voxside; ++z)
                {
                    //
                    // let's look if we need to bump up the resolution (when vp0special is passed)
                    bool bfound = false;
                    if (vp0special != nullptr)
                        for (auto it : *vp0special)
                            if (x == std::get<0>(it) && y == std::get<1>(it) && z == std::get<2>(it))
                                bfound = true;
                    // let's go high resolution for the voxel specified in the vector.
                    if (bfound)
                    {
                        int vsl = voxelSideAtLevel[maxlevel - level];
                        int maxvsl = voxelSideAtLevel[maxlevel];
                        for (int sx = 0; sx < vsl; sx++)
                            for (int sy = 0; sy < vsl; sy++)
                                for (int sz = 0; sz < vsl; sz++)
                                {
                                    int xvp0 = vsl*x + sx;
                                    int yvp0 = vsl*y + sy;
                                    int zvp0 = vsl*z + sz;
                                    processvoxel(maxlevel, xvp0, yvp0, zvp0); // need to receive vp0 coordinate
                                }
                    }
                    else {
                        if (allres)
                        {
                            Eigen::Vector4d pvp0 = vp->B[level] * Eigen::Vector4d{ static_cast<double>(x), static_cast<double>(y), static_cast<double>(z), 1.0 };
                            int xvp0 = static_cast<int>(pvp0[0]);
                            int yvp0 = static_cast<int>(pvp0[1]);
                            int zvp0 = static_cast<int>(pvp0[2]);
                            if (processvoxel(level, xvp0, yvp0, zvp0) && StopAtFirstFace) // need to receive vp0 coordinate
                                break; // this break will only leave the first face in the ply, not the back face
                        }
                    }
                }
            }
        }
#ifndef SINGLE_THREAD_PLY
        );
#endif
        if (myfunc._Empty() == false)
        {
            // if callback is present, call it so that custom drawing can happen
            myfunc(level, listofpos, triangles);
        }
    

        //dispDuration("    mask processed");

        std::ofstream ofs;
        ofs.open(filename.c_str(), std::ofstream::out);
        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "comment mview5 by Marc Racicot\n";

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        ofs << "comment " << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << "\n";
        ofs << "element vertex " << listofpos.size() << "\n";
        ofs << "property float32 x\n";
        ofs << "property float32 y\n";
        ofs << "property float32 z\n";
        ofs << "property uint8 red\n";
        ofs << "property uint8 green\n";
        ofs << "property uint8 blue\n";
        ofs << "element face " << triangles.size() << "\n";
        ofs << "property list uint8 uint32 vertex_indices\n"; // { "vertex_indices" is a list of ints }
        ofs << "end_header\n";

        std::stringstream tempstr;
        for (auto& i : listofpos)
        {
            char line[200];
            if (i.level == -1) // already in world coordinates
            {
                sprintf_s(line, sizeof(line), "%f %f %f %d %d %d\n", i.x, i.y, i.z, int(255), int(255), int(255));
            }
            else
            {
                Eigen::Vector4d pw = getWorldPosition(i.level, m_viewpointsvec[0], i.x, i.y, i.z);
                // I know the viewpoint id, so I can retreive the best camera match from the vp cam set.
                // using this camera and the worldposition, let's retreive a color
                //
                if (i.vpid != -1)
                {
                    int imgnum = std::get<0>(m_viewpointsvec[i.vpid]->cameraset[0]);
                    Eigen::Vector2d pse = getScreenPosition(pw, imgnum);
                    float rgbv[4]{128, 128, 128, 128};
                    if (bilinear(pse, m_imgsvec[imgnum]->image, 0.0f, rgbv)){
                        sprintf_s(line, sizeof(line), "%f %f %f %d %d %d\n", pw[0], pw[1], pw[2], int(rgbv[0] * 255), int(rgbv[1] * 255), int(rgbv[2] * 255));
                    }
                    else{
                        sprintf_s(line, sizeof(line), "%f %f %f %d %d %d\n", pw[0], pw[1], pw[2], int(255), int(0), int(0));
                    }
                }
                else
                {
                    sprintf_s(line, sizeof(line), "%f %f %f %d %d %d\n", pw[0], pw[1], pw[2], int(128), int(128), int(128));
                }
            }
            ofs << line;
        }
        ofs.clear();
        for (auto& t : triangles)
        {
            char line[200];
            sprintf_s(line, sizeof(line), "3 %d %d %d\n", t.t1, t.t2, t.t3);
            ofs << line;
        }
        ofs.flush();
        ofs.close();
}
