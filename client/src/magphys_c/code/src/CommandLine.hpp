/*
 *    (c) UWA, The University of Western Australia
 *    M468/35 Stirling Hwy
 *    Perth WA 6009
 *    Australia
 *
 *    Copyright by UWA, 2012-2013
 *    All rights reserved
 *
 *    This library is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *    MA 02111-1307  USA
 */

#include <string>
using std::string;

#include <vector>
using std::vector;

#ifndef COMMAND_LINE_HPP
#define COMMAND_LINE_HPP

namespace magphys {

class CommandLine {
public:
	CommandLine();
	~CommandLine();

	bool loadArguments(const vector<string>& args);
	inline string observationsFile() const {
		return observationsFile__;
	}
	inline string filtersFile() const {
		return filtersFile__;
	}
	inline string modelOpticalFile() const {
		return modelOpticalFile__;
	}
	inline string modelInfraredFile() const {
		return modelInfraredFile__;
	}
	inline double redshift() const {
		return redshift__;
	}
    inline int startingLine() const {
        return startingLine__;
    }

private:
	bool checkFiles();

	double redshift__ = 0;
	string observationsFile__;
	string filtersFile__;
	string modelOpticalFile__;
	string modelInfraredFile__;
	int startingLine__ = 0;
};

}

#endif
