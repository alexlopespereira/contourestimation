
#ifndef GENERIC_TESTCASE_H
#define GENERIC_TESTCASE_H
#include "OOTestCase.h"
#include <string>

using namespace std;

namespace Util {

class GenericTestCase : public OOTestCase
{
public:

	GenericTestCase (string srcdir, string outdir, string testname, string extension, int fnl, string prefix, int firstFileOption):OOTestCase(srcdir, outdir, testname, extension, fnl, prefix, firstFileOption){	};

	virtual ~GenericTestCase ( ) { }

	void setup(){	}

	Mat next(){}

protected:

private:

};
}; // end of package namespace

#endif //
