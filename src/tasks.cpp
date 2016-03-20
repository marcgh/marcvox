// +++++++++++++++++++++++
//  Marc Racicot  @ 2014
// +++++++++++++++++++++++

#include "tasks.h"
#include "tbb/tbb.h"
#include "tbb/blocked_range.h"
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <regex>

using std::string;

template<typename F>
class lambda_task : public tbb::task {
   F my_func;
   /*override*/ tbb::task* execute( ) {
      my_func( );
      return NULL;
   }
public:
   lambda_task( const F& f ) : my_func( f ) {}
};

template<typename F>
void tbb_enqueue_lambda( const F& f ) {
   tbb::task::enqueue( *new(tbb::task::allocate_root( )) lambda_task<F>( f ) );
}

int imgproTasks::addTask( const std::vector<string>& v)
{
   taskslist.push_back( v );
   return 0;
}

void removeSpaces(std::string & str)
{
    size_t position = 0;
    for (position = str.find(" "); position != std::string::npos; position = str.find(" ", position))
    {
        str.replace(position, 1, "%20");
    }
}

void insertSpaces(std::string & str)
{
    size_t position = 0;
    for (position = str.find("%20"); position != std::string::npos; position = str.find("%20", position))
    {
        str.replace(position, 3, " ");
    }
}

std::list<std::string> imgproTasks::gettextoutput( )
{
   std::list<std::string> textoutput;
   for ( auto i : taskslist )
   {
      int cnt = 0;
      std::string line;
      for ( auto j : i )
      {
         removeSpaces(j);
         if ( cnt == 1 )
         {
            line += "(";
         }

         if ( j.size( ) <= 0 )
            break;

         if ( cnt == 0 )
            line.append( j );
         else
         {
            std::string tmp;
            if ( cnt > 1 )
               tmp = ", ";
            else
               tmp = " ";
            tmp += "\"" + j + "\"";
            line.append( tmp );
         }
         ++cnt;
      }
      if (cnt > 1)
      {
          line += " )";
      }
      line += ";\n";
      textoutput.push_back(line);
   }
   return textoutput;
}

void imgproTasks::displayTasks()
{
   std::list<std::string> txtout = gettextoutput( );
   for ( auto i : txtout )
   {
      OutputDebugString( i.c_str() );
   }
}

int imgproTasks::saveTaskFile( std::string filename )
{
   std::list<std::string> txtout = gettextoutput( );
   std::ofstream out_file( filename );
   for ( auto i : txtout )
      out_file << i;
   out_file.close( ); 
   return 0;
}

int imgproTasks::loadTaskFile( std::string filename )
{
   std::ifstream in_file( filename );
   string line;
   taskslist.clear();
   if ( in_file.is_open( ) )
   {
      while ( getline( in_file, line ) )
      {
         std::regex re( "[\\s,();\"]+" );
         std::sregex_token_iterator i( line.begin( ), line.end( ), re , -1);
         std::vector<std::string> vecofstrings( i, std::sregex_token_iterator() );
         for (auto &j : vecofstrings)
             insertSpaces(j);  // %20 --> ' '
         taskslist.push_back( vecofstrings );
      }
      in_file.close( );
   }
   return 0;
}

void imgproTasks::executeTasks( )
{
    for (auto t : taskslist)
    {
        try {
            // to do use iterator begin...current to add in ply file
            if (executeTask(t) == false)
                break;
        }
        catch (...)
        {
            // exception error ... feedback please
        }
   }
}


bool imgproTasks::executeTask( const std::vector<std::string>& vst )
{
   // nothing done in base class
    return true;
}