// +++++++++++++++++++++++
//  Marc Racicot  @ 2014
// +++++++++++++++++++++++
#pragma once

#include <string>
#include <list>
#include <vector>
#include <chrono>
#include <sstream>
#include <iostream>

#undef min
#undef max

//
// This class is used to manage the sequence of commands (or tasks) that will be executed
//
class imgproTasks
{
public:
   imgproTasks() {};

   int addTask( const std::vector<std::string>& v);

   void displayTasks();
   int saveTaskFile( std::string filename );
   int loadTaskFile( std::string filename );

   std::list<std::vector<std::string>> taskslist;

   virtual bool executeTask(const std::vector<std::string>& vst);
   void executeTasks( );
   void clearTasks() { taskslist.clear(); }

private:
   std::list<std::string> gettextoutput( );
};



static inline void dispDuration(
    const std::string& str = "",
    bool display = true)
{
   static std::chrono::time_point<std::chrono::system_clock> t0;
   std::chrono::time_point<std::chrono::system_clock> t1;
   if (str == "" )
   {
      t0 = std::chrono::high_resolution_clock::now( );
      return;
   }
   t1 = std::chrono::high_resolution_clock::now( );

   std::stringstream buffer;
   float fduration1 = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count( );
   fduration1 /= 1000.0;
   buffer << str;
   if (display)
   {
       buffer << ":" << fduration1 << " sec  ";
   }
   buffer << "\n";
   std::cout << buffer.str().c_str();
   t0 = t1;
}

