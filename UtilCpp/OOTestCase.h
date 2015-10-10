
#ifndef OOTESTCASE_H
#define OOTESTCASE_H


#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
//#include "UtilCpp.h"
#include <opencv2/opencv.hpp>
enum first_file_options{NUMBER_1=1, TEMPORAL_ROI=10};
using namespace cv;
using namespace std;

namespace Util {
class OOTestCase
{
public:
//	OOTestCase(){}
	OOTestCase(string extn=string("bmp"), string outpath=("./"))
	{
		ext = extn;
	}

	OOTestCase(string srcdir, string outdir, string testname, string extension, int fnl, string prefix, int firstFileOption)
	{
		ext.append(extension);
		string roiStr(srcdir+"/ROI.jpg");
		roi = imread(roiStr);
		if(roi.cols==0)
		{
			char fname[10];
			sprintf(fname, "%06d.", firstFileOption);
			string path0 = srcdir+"/"+prefix+string(fname)+extension;
			roi=imread(path0);
		}
		seqname = testname;
		w=roi.cols;
		h=roi.rows;
		wh=w*h;
		fnlength=fnl;
		string str;
		ifstream infile;
		string f = string(srcdir + "/temporalROI.txt");
		infile.open(f.c_str());
		getline(infile,str); // Saves the line in STRING.
		infile.close();
		string first = str.substr(0, str.find_last_of(" "));
		stringstream firststream(first); //create the stringstream
		if(firstFileOption>=0)
		{
			firstfile=firstFileOption;
			lastfile=100000;
		}
		else if(firstFileOption==TEMPORAL_ROI)
			firststream>>firstfile; //convert the string to an integer

		string num = str.substr(str.find_last_of(" ") + 1);
		stringstream myStream(num); //create the stringstream
		myStream>>lastfile; //convert the string to an integer
		lastfile+=10;
		channels=3;
		imgsize.width=w;
		imgsize.height=h;
		//firstfile=1;
		seqname.append(testname);
		path.append(srcdir + "/" + prefix);
		gtpath.append(srcdir + "/../groundtruth/gt");//.assign("/home/alex/TestData/cdw/dataset/baseline/highway/groundtruth/gt");
		outpath.append(outdir + "/bin");//.assign("/home/alex/TestData/cdw/results/baseline/highway/");
	}

	virtual ~OOTestCase(){}
	virtual void setup ( )=0;
    int getChannels() const;
    void setChannels(int channels);
    Size getImgsize() const;
    void setImgsize(Size imgsize);

    string createFilename(string & path, int number, int characters, string & ext)
    {
    	string result(path);
    	char ns[6];
    	if(characters==6)
			sprintf(ns, "%06d.", number);
		else if(characters==5)
    		sprintf(ns, "%05d.", number);
    	else if(characters==4)
    		sprintf(ns, "%04d.", number);
    	else if(characters==3)
    		sprintf(ns, "%03d.", number);
    	else
    		fprintf(stderr, "\nWrong number of characters to create file name.");

    	result.append(ns);
    	result.append(ext);
    	return result;
    }

    Vec4i countErrors(Mat & gt, Mat & pic)
    {
    	Vec4i cmp;
    	int i, l, w, h;
    	cmp.val[0]=0;
    	cmp.val[1]=0;
    	cmp.val[2]=0;
    	cmp.val[3]=0;
    	int gtfg=0, picfg=0;
    	h=gt.rows;
    	w=gt.cols;
    	Mat_<uchar>& gtImg = (Mat_<uchar>&)gt;
    	Mat_<uchar>& picImg = (Mat_<uchar>&)pic;
    	for(i=0; i<h;i++)
    	{
    		for(l=0; l<w;l++)
    		{
    			if(gtImg(i,l)==0)
    			{
    				if(picImg(i,l)==255)
    				{
    					picfg++;
    					//r.fp++;
    					cmp.val[1]++;
    				}
    				else if(picImg(i,l)==0)
    					//r.tn++;
    					cmp.val[3]++;
    				else
    					cerr << "unknown value: " << picImg(i,l) << endl;
    			}
    			else
    			{
    				gtfg++;
    				if(picImg(i,l)==0)
//    					r.fn++;
    					cmp.val[0]++;
    				else if(picImg(i,l)==255)
    				{
    					picfg++;
//    					r.tp++;
    					cmp.val[2]++;
    				}
    				else
    					cerr << "unknown value: " << picImg(i,l) << endl;
    			}
    		}
    	}
    	return cmp;
    }
    Mat getGT(int frn)
	{
		string ext("bmp");
		string file = createFilename(gtpath, frn, fnlength, ext);

		Mat gt = imread(file);
		if(gt.dims>1)
		{
			vector<Mat> planes;
			split(gt, planes);
			gt=planes[0];
		}
//		else if(gt.dims<1)
//		{
//			cerr << "unexpected image dimension: " << gt.dims << endl;
//		}
		return gt;
	}

    virtual Mat next()=0;

    Vec4f evaluate(Mat & src, int frn)
    {
    	Mat gt = getGT(frn);
		Vec4f r = countErrors(gt, src);
		return r;
    }

	string getOutpath() const {
		return outpath;
	}

	void setOutpath(string outpath) {
		this->outpath = outpath;
	}

	string getExt() const {
		return ext;
	}

	void setExt(string ext) {
		this->ext = ext;
	}
    unsigned short getGtlength() const
    {
        return gtlength;
    }

    int getTestset() const
    {
        return testset;
    }

    void setTestset(int testset)
    {
        this->testset = testset;
    }

    const string * getBmphder() const
    {
        return & bmphder;
    }

    int getFirstfile() const
    {
        return firstfile;
    }

    int getFnlength() const
    {
        return fnlength;
    }

    unsigned int getGTContent(int i) const
    {
        return GT[i];
    }

    const string * getGtpath() const
    {
        return &gtpath;
    }

    int getH() const
    {
        return h;
    }

    int getLastfileNumber() const
    {
        return lastfile;
    }

    int getNfiles() const
    {
        return lastfile -firstfile + 1;
    }

    string getPath() const
    {
        return path;
    }

    string getSeqname() const
    {
        return seqname;
    }

    string getTestname() const
    {
        return testname;
    }

    string getTestname_all() const
    {
        return testname_all;
    }

    string getTestname_sdelta() const
    {
        return testname_sdelta;
    }

    int getW() const
    {
        return w;
    }

    int getWh() const
    {
        return wh;
    }
    int getChannels()
    {
        return channels;
    }

    Size getImgsize()
    {
        return imgsize;
    }

protected:

  int w;
  int h;
  int wh;
  int firstfile;
  int fnlength;
  int nfiles;
  int lastfile;
  int channels;
  Mat roi;
  Size imgsize;
  string ext;
  string bmphder;
  string testname;
  string seqname;
  string testname_all;
  string testname_sdelta;
  string path;
  string gtpath;
  string outpath;
  unsigned short GT[200];
  int testset;
  unsigned short gtlength;

private:


};
};


 // end of package namespace

#endif // TESTCASE_H
