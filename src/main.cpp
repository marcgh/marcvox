// +++++++++++++++++++++++
//  Marc Racicot  @ 2015
// +++++++++++++++++++++++
#include <stdio.h>
#include <iostream>
#include "imgprocess.h"
#include <conio.h>
#include <string>
#include "tbb/task_scheduler_init.h"
#include <tclap/CmdLine.h>
#include <Eigen/Dense>  
//#define GLOG_NO_ABBREVIATED_SEVERITIES

int main(int argc, char *argv[])
{
    std::string filename{};
    int nbthread{ -1 };

    // Command line parser!
    try {
        TCLAP::CmdLine cmd("marcvox", ' ', "1.0");
        TCLAP::ValueArg<std::string> filenameArg("f", "filename", "filename to process", true, "temple.tsk", "string");
        cmd.add(filenameArg);

        TCLAP::ValueArg<int> itest("t", "thread", "number of thread for intel tbb", false, 4, "int");
        cmd.add(itest);

        cmd.parse(argc, argv); // Parse the argv array.

        // Get the value parsed by each arg. 
        filename = filenameArg.getValue();
        std::cout << "Will process filenname \"" << filename << "\"" << std::endl;
        if (itest.isSet())
        {
            std::cout << "intel tbb nbthread: " << itest.getValue() << std::endl;
            nbthread = itest.getValue();
        }
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    if (nbthread > 0)
    {
        tbb::task_scheduler_init init(nbthread);
    }
    if (filename.size() > 0)
    {
        try {
            ImageProcessing imgpro;
            imgpro.loadTaskFile(filename);
            imgpro.executeTasks();
        }
        catch (...) {
            std::cout << "unhandled exception catched.";
            exit(0);
        };
    }
    else
    {
        std::cout << "Nothing to do !\n";
    }
    return 0;
}

