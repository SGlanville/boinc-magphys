The wrapper code needs to be built in the boinc directory structure.

*Linux*

For Linux simply compile the <boinc>/samples/wrapper code and job done.

*Windows*

Windows is much more complex.
1. You need MinGW installed
2. Go to <boinc>/lib and make using Makefile.mingw - this will build the libraries required
3. Copy the file Makefile.mingw in this wrapper to /boinc/samples/wrapper/Makefile.mingw and build the file build the file
