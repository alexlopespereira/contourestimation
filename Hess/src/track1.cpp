/*
  Perform single object tracking with particle filtering

  @author Rob Hess
  @version 1.0.0-20060306
*/

#include "defs.hpp"
#include "utils.hpp"
#include "particles.h"
#include "observation.h"
#include "UtilCpp.h"
//#include "Hess.h"
#include "HessTracker.h"
#include "HessParticle.h"
//#include "FragHessTracker.h"
#include "Enums.h"
#include <vector>
#include <string>
#include <cstring>

//#include "Evaluation/cdw/CDW_Highway.h"

using namespace std;
using namespace Util;
using namespace HessTracking;

int num_particles;
BackgroundSubtractorMOG2 pMOG; //MOG Background subtractor


/***************************** Function Prototypes ***************************/
extern "C" {
void getSusanEdges(unsigned char * buffer, int t, int x_size, int y_size);
}

string findSegmentationName(string path)
{
	char *pch;
	char strchar[150];
	strcpy(strchar, path.c_str());

	pch = std::strtok(&strchar[0],"/");
	string token(pch);

	while (token.compare(string("results")) != 0 && pch!=NULL)
	{
		pch = strtok (NULL, "/");
		if(pch!=NULL)
			token = string(pch);
	}
	pch = strtok (NULL, "/");
	if(pch!=NULL)
		token = string(pch);
	return token;
}

string findLastToken(string path)
{
	int tries=0;
	char *pch;
	char strchar[150];
	strcpy(strchar, path.c_str());
	string lasttoken;
	pch = std::strtok(&strchar[0],"/");
	string token(pch);

	while (tries<2)
	{
		if(pch!=NULL)
			lasttoken = string(token);

		pch = strtok (NULL, "/");
		if(pch!=NULL)
			token = string(pch);
		else
			tries++;
	}
	return lasttoken;
}

void testPropMedian(Mat & mask)
{
	UtilCpp u;
	Mat med = u.proportionalMedianBlur(mask);
	imshow("mask",mask);
	if(!med.empty())
		imshow("med",med);
	waitKey(0);
}

void denoiseImage(Mat src, string outpath, int currframe)
{
	Mat out;
	fastNlMeansDenoisingColored(src,out);
	UtilCpp u;
	u.writeImg(outpath,out,currframe,"bmp");
}

void getFeatures(string inputpath, string maskpath, string outpath, string inputPrefix, string extinput, int firstFileP, int lastfile, int step)
{
	GhostDetector gd;
	vector< vector<int> > vflist;
	UtilCpp u;
#ifndef ALL_FEATURES
	string csvfile="/home/alex/Desktop/alp/doutorado/05.Implementation/11.Thirdparty/Hess/testHeader.csv";
	vflist = u.getFeatureList(csvfile);
#endif
#ifdef EVAL_SVM
	gd.loadSvmModel(string("svm_modelExp29NoK.xml"));//svm_modelExp29NoK.xml //svm_modelExp32.xml
#endif
    int currFrame=12;
	int lastFrame=lastfile;
	string f = string(inputpath + "/temporalROI.txt");
	string maskprefix("/bin");
	string extmask("png");
	int count=0;
	std::ofstream ofs;
	string featuresFile=outpath+"/features.csv";
	string headerFile=outpath+"/header.csv";
	int totalNonBlack=0;
	int totalGhost=0;
	int totalNoGhost=0;
	int totalAnalyzed=0;

#ifdef TRAINING
#ifdef CREATE_HEADER
	ofs.open(headerFile.c_str(), std::ofstream::out | std::ofstream::trunc);
	ofs.close();
#else
	ofs.open(featuresFile.c_str(), std::ofstream::out | std::ofstream::trunc);
	ofs.close();
#endif
#endif
	string str;
	ifstream infile;
	infile.open(f.c_str());
	if(!infile.good())
	{
		//cerr << "could not open file " << f << endl;
		//exit -1;
		currFrame=1;
		lastFrame=100000;
	}
	else
	{
		getline(infile,str); // Saves the line in STRING.
		infile.close();
		string first = str.substr(0, str.find_last_of(" "));
		stringstream firststream(first); //create the stringstream
		firststream>>currFrame; //convert the string to an integer
		//currFrame+=11;

		string num = str.substr(str.find_last_of(" ") + 1);
		stringstream myStream(num); //create the stringstream
		myStream>>lastFrame; //convert the string to an integer
#ifdef DENOISE
		currFrame=1;
#endif
	}
	gd.setOutpath(outpath);
	gd.setExtinput(extinput);
	gd.setInputpath(inputpath);
	gd.setInputPrefix(inputPrefix);
	gd.setMaskpath(maskpath);
	gd.setExtmask(extmask);
	gd.setMaskprefix(maskprefix);
	gd.setSegmentationName(findSegmentationName(maskpath));
	gd.setTestCaseName(findLastToken(maskpath));
	Mat filteredBG, filteredRGB;

#ifdef ASOD
	string bgfile(inputpath+"/background.png");
	Mat bg=imread(bgfile);
	bilateralFilter(bg, filteredBG, 13, 50, 20);//25, 50, 15
	gd.setMinArea(50);
	if(firstFileP>=0)
		currFrame=firstFileP;
	if(lastfile>0)
		lastFrame=lastfile;
	gd.setFrameNo(currFrame);
#else
	Mat bg = u.getFrame(false,maskpath+"/bg/bg",currFrame,6,"bmp");
	filteredBG=bg;
	gd.setMinArea(300);
	//gd.initAccEdges(bg);
	Mat input = u.getFrame(false, inputpath+inputPrefix, currFrame, 6, extinput);
	//gd.initAccEdges(input);
#endif

	Mat tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, 6, extmask);
	if(tmpMask.cols==0)
	{
		if (extmask.compare("jpg") == 0)
		{
			extmask=string("png");
			tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, 6, extmask);
		}
		else
		{
			extmask=string("jpg");
			tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, 6, extmask);
		}
	}
	gd.setWorld(tmpMask);
	while( currFrame <= lastFrame )
	{
		Mat mask = u.getFrame(false, maskpath+maskprefix, currFrame, 6, extmask);
		Mat input = u.getFrame(false, inputpath+inputPrefix, currFrame, 6, extinput);
#ifdef DENOISE
		denoiseImage(input,inputpath+"/denoise/de",currFrame);
		currFrame+=step;
		continue;
#endif
#ifndef ASOD
		bg = u.getFrame(false,maskpath+"/bg/bg",currFrame,6,"bmp");
		filteredBG=bg;
//		bilateralFilter(bg, filteredBG, 15, 40, 20);
#endif
		if(mask.cols==0)
		{
			currFrame+=step;
			gd.setFrameNo(currFrame);
			continue;
		}
		Mat maskf=mask;
//		Mat inputf=input;
#ifdef  ASOD
		if(input.empty())
		{
			input = u.getFrame(false, inputpath+"/input/in", currFrame, 6, "png");
			bilateralFilter(input, filteredRGB, 15, 40, 20);
			u.writeImg(inputpath+"/filtered/fi",filteredRGB,currFrame,"bmp");
		}
		else
			filteredRGB=input;
#else
		bilateralFilter(input, filteredRGB, 5, 12, 5);
#endif
		Mat resultMask = Mat::zeros(maskf.size(),CV_8U);
		Mat mask8b;
		if(maskf.channels()==3)
			cvtColor(maskf,mask8b,CV_BGR2GRAY);
		else
			mask8b=maskf;
		threshold( mask8b, mask8b, 180, 255, THRESH_BINARY );

		//gd.detectFrameGhosts(inputf,mask8b, vflist);
		vector<vector<float> > bgfeatures, infeatures;
		Mat mtemp = mask8b.clone();
		vector<int> areas;
		vector<vector<Point> > analyzedContours;
		vector<Rect> rects;
		//Melhor combinacao dos parametros do histograma f3,b1,m3,g1,i3,j2,k1,l0
		bgfeatures = gd.getFrameFeatures(filteredBG,mtemp,vflist,areas, analyzedContours, rects);
		infeatures = gd.getFrameFeatures(filteredRGB,mask8b,vflist,areas, analyzedContours, rects);
		uint lp;

		if(bgfeatures.size()>0)
			totalNonBlack++;
		for(lp=0; lp<bgfeatures.size(); lp++)
		{
			vector<float> sub = u.subtract(bgfeatures[lp], infeatures[lp]);
//			cout <<"bg:" << bgfeatures[lp][0] << ", " << bgfeatures[lp][1] << endl;
//			cout <<"in:" << infeatures[lp][0] << ", " << infeatures[lp][1] << endl;
//			cout << areas[lp] << "," << sub[0] << ", " << sub[1] << endl;
//			cout << sub[0] << ", " << infeatures[lp][0] << ", " << bgfeatures[lp][0] << endl;
			u.printVector(sub, true);
//			if(areas[lp]>300)
//			{
//				if(sub[0]+sub[1]<0)
//					totalGhost++;
//				else
//					totalNoGhost++;
//			}
//			else
//			{
				if(sub[0]<0)
				{
					totalGhost++;
#ifdef REPORT_RESULT
					//drawContours(fixedMask,analyzedContours,lp,Scalar(0),-1);
//#ifdef DEBUG
					Mat tmpc = filteredRGB.clone();
					drawContours(tmpc,analyzedContours,lp,Scalar(0,0,255),2);
					u.drawText(tmpc,TITLE,"subh",sub[0],Point(10,10));
					u.drawText(tmpc,SUB_TITLE,"subc",sub[1],Point(10,80));
					u.writeImg(outpath+"/ghostfeed/in",tmpc,100*currFrame+lp,"png");
//#endif
#endif
				}
				else
				{
					totalNoGhost++;
				}
//			}
			totalAnalyzed++;
		}
#ifdef REPORT_RESULT
		//u.writeImg(outpath+"/ghost/bin",fixedMask,currFrame,"png");
#endif
		Mat smallmask;
#ifdef DEBUG
		if(mask8b.rows>800 || mask8b.cols>800)
			resize(mask8b, smallmask, Size(mask8b.cols/2,mask8b.rows/2), 0, 0, INTER_LINEAR);
		else
			smallmask=mask8b;
		imshow("mask", smallmask);
		waitKey( 1 );
#endif
		input.release();
		currFrame+=step;
		gd.setFrameNo(currFrame);
	}


	cout << inputpath << "," << totalNonBlack << "," << totalAnalyzed << "," << totalGhost << "," << totalNoGhost << endl;
}

void testHessTracker(string inputpath, string maskpath, string outpath, string inputPrefix, string extinput, int firstFileP, int lastfile, int step)
{
	HessTracker hesst;
    num_particles=20;
	//GhostDetector gd;
	int i=0;
	UtilCpp u;
	vector< vector<int> > vflist;
#ifndef ALL_FEATURES
	string csvfile="/home/alex/Desktop/alp/doutorado/05.Implementation/11.Thirdparty/Hess/testHeader.csv";
	vflist = u.getFeatureList(csvfile);
	hesst.setVflist(vflist);
#endif
#ifdef EVAL_SVM
	gd.loadSvmModel(string("svm_modelExp29NoK.xml"));//svm_modelExp29NoK.xml //svm_modelExp32.xml
#endif
#ifdef EVAL_NAIVE_BAYES
	hesst.loadBayesModel(string("/home/alex/Desktop/alp/doutorado/05.Implementation/11.Thirdparty/Hess/bayes.yaml"));//svm_modelExp29NoK.xml //svm_modelExp32.xml
#endif
    int currFrame=-1;
	int lastFrame=lastfile;
	string f = string(inputpath + "/temporalROI.txt");
	string maskprefix("/bin");
	string extmask("png");
	std::ofstream ofs;
	string featuresFile=outpath+"/features.csv";
	string headerFile=outpath+"/header.csv";

#ifdef CREATE_HEADER
	ofs.open(headerFile.c_str(), std::ofstream::out | std::ofstream::trunc);
	ofs.close();
#else
	ofs.open(featuresFile.c_str(), std::ofstream::out | std::ofstream::trunc);
	ofs.close();
#endif

	string str;
	ifstream infile;
	infile.open(f.c_str());
	if(!infile.good())
	{
		if(firstFileP>=0)
			currFrame=firstFileP;
		else
			currFrame=1;
		if(lastfile>0)
			lastFrame=lastfile;
	}
	else
	{
		getline(infile,str); // Saves the line in STRING.
		infile.close();
		string first = str.substr(0, str.find_last_of(" "));
		stringstream firststream(first); //create the stringstream
		firststream>>currFrame; //convert the string to an integer
		if(currFrame>=2)
			currFrame-=2;

		string num = str.substr(str.find_last_of(" ") + 1);
		stringstream myStream(num); //create the stringstream
		myStream>>lastFrame; //convert the string to an integer
	}
	hesst.setOutpath(outpath);
	hesst.setExtinput(extinput);
	hesst.setInputpath(inputpath);
	hesst.setInputPrefix(inputPrefix);
	hesst.setMaskpath(maskpath);
	hesst.setExtmask(extmask);
	hesst.setMaskprefix(maskprefix);
	hesst.setSegmentationName(findSegmentationName(maskpath));
	hesst.setTestCaseName(findLastToken(maskpath));
	Mat filteredBG, filteredRGB;
	int fnlen=6;
#ifdef ASOD
	string bgfile(inputpath+"/background.png");
	Mat bg=imread(bgfile);
	bilateralFilter(bg, filteredBG, 13, 50, 20);//25, 50, 15
	//hesst.setMinArea(50);
	hesst.setFrameNo(currFrame);
#else
	Mat bg = u.getFrame(false,maskpath+"/bg/bg",currFrame,fnlen,"bmp");
	filteredBG=bg;
//	gd.setMinArea(300);
	//gd.initAccEdges(bg);
	Mat input = u.getFrame(false, inputpath+inputPrefix, currFrame, fnlen, extinput);
	//gd.initAccEdges(input);
#endif

	Mat tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, fnlen, extmask);
	if(tmpMask.cols==0)
	{
		if (extmask.compare("jpg") == 0)
		{
			extmask=string("png");
			tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, fnlen, extmask);
		}
		else
		{
			extmask=string("jpg");
			tmpMask = u.getFrame(false, maskpath+maskprefix, currFrame, fnlen, extmask);
		}
	}
	//gd.setWorld(tmpMask);
	hesst.setMinArea(tmpMask,600);
//	cout << inputpath << endl;
	while( currFrame <= lastFrame )
	{
		Mat mask = u.getFrame(false, maskpath+maskprefix, currFrame, fnlen, extmask);
		Mat input = u.getFrame(false, inputpath+inputPrefix, currFrame, fnlen, extinput);
#ifndef ASOD
		bg = u.getFrame(false,maskpath+"/bg/bg",currFrame,fnlen,"bmp");
		filteredBG=bg;
//		bilateralFilter(bg, filteredBG, 15, 40, 20);
#endif
		if(mask.cols==0)
		{
			currFrame+=step;
			hesst.setFrameNo(currFrame);
			continue;
		}
		Mat maskf=mask;
//		Mat inputf=input;
#ifdef  ASOD
		if(input.empty())
		{
			input = u.getFrame(false, inputpath+"/input/in", currFrame, fnlen, "png");
			bilateralFilter(input, filteredRGB, 15, 40, 20);
			u.writeImg(inputpath+"/filtered/fi",filteredRGB,currFrame,"bmp");
		}
		else
			filteredRGB=input;
#else
		bilateralFilter(input, filteredRGB, 5, 12, 5);
#endif
		Mat resultMask = Mat::zeros(maskf.size(),CV_8U);
		Mat mask8b;
		if(maskf.channels()==3)
			cvtColor(maskf,mask8b,CV_BGR2GRAY);
		else
			mask8b=maskf;
		threshold( mask8b, mask8b, 180, 255, THRESH_BINARY );

		if( i == 0 )
			hesst.init(filteredRGB,mask8b,filteredBG,num_particles);
		else
			hesst.next(filteredRGB,mask8b,resultMask,filteredBG);

#ifdef DEBUG
		Mat inputfc = filteredRGB.clone();
		hesst.showResults(inputfc);

		Mat smallin, smallbg;
		if(inputfc.rows>600 || inputfc.cols>600)
		{
			resize(inputfc, smallin, Size(inputfc.cols/3,inputfc.rows/3), 0, 0, INTER_LINEAR);
			resize(bg, smallbg, Size(bg.cols/3,bg.rows/3), 0, 0, INTER_LINEAR);
		}
		else
		{
			smallin=inputfc;
			smallbg=bg;
		}

		u.drawText(smallin,SUB_TITLE,string(""),currFrame);
		imshow("Input", smallin);
		imshow("Background",smallbg);

		cout.flush();
		//u.writeJPG("result/tracking", input, currFrame);
		if(currFrame==firstFileP)
			waitKey(0);
		else
			waitKey( 1 );
#endif
//		if(currFrame%100==0)
//			cout << endl << "Analised frame " << currFrame;

		input.release();
		i++;
		currFrame+=step;
		hesst.setFrameNo(currFrame);
	}
	hesst.analyseRemainingTrajectories();
//	hesst.printCSVline();
//	hesst.writeMapToFile();
	int totalNonBlack = hesst.getTotalNonBlack();
	int totalGhost = hesst.getTotalGhosts();
	int totalAnalyzed = hesst.getTotalAnalyzed();
	double rate=totalGhost/(double)totalNonBlack;
	double rateAn=totalGhost/(double)totalAnalyzed;

	cout << inputpath << ", " << totalNonBlack << ", " << totalAnalyzed << "," << totalGhost << "," << totalAnalyzed-totalGhost << endl;
//    std::cout.precision(5);
//	cout << rate << "," << rateAn << endl;
//	cout << r1 << "," << r2 << "," << r3 << "," << r4 << "," << r5 << "," << r6 << "," << r7 << endl;
}

Mat processImages(Mat frame, int frameNumber) {
  //read the first file of the sequence
Mat fgMaskMOG; //fg mask generated by MOG method

    //update the background model
    pMOG(frame, fgMaskMOG);
    UtilCpp u;
    string number = u.intToString(frameNumber);
    rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
              cv::Scalar(255,255,255), -1);
    putText(frame, number.c_str(), cv::Point(15, 15),
            FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
    //show the current frame and the fg masks
    imshow("Frame", frame);
    imshow("FG Mask MOG", fgMaskMOG);
    waitKey(1);
    return fgMaskMOG;
}

void testBS(string inputpath, string maskpath, string inputPrefix, string extinput, int firstFileP, int lastfile, int step)
{

    int currFrame=-1;
	int lastFrame=lastfile;
	string maskprefix("/bin");
	string extmask("png");
	std::ofstream ofs;
	UtilCpp u;
	string str;

	if(firstFileP>=0)
		currFrame=firstFileP;
	else
		currFrame=1;
	if(lastfile>0)
		lastFrame=lastfile;

	while( currFrame <= lastFrame )
	{
		Mat input = u.getFrame(false, inputpath+inputPrefix, currFrame, 6, extinput);
		Mat resultFg = processImages(input,currFrame);
		string rname = string(maskpath)+"/bin";
		string bgname = string(maskpath)+"/bg/bg";
		u.writeImg(rname, resultFg, currFrame, string("jpg"));
		Mat bg;
		pMOG.getBackgroundImage(bg);
		u.writeImg(bgname,bg,currFrame,string("bmp"));
		input.release();
		currFrame+=step;
	}
}

void testMedian()
{
	HessParticle p;
	float tmp[] = { 10, 10, 11, 20, 30, 100 };
	vector<float> v( tmp, tmp+6 );
	cout << p.median(v) << endl;
}

void testSusan()
{
	GhostDetector gd;
	UtilCpp u;
	Mat img = imread("/home/alex/TestData/ASODds/dataset/C2/C2_All_stolen_anot/EPFL_indoor_activity_stolen_object_cif_seg1_xvid/background.png");
	Mat img8b;
	cvtColor(img, img8b, CV_BGR2GRAY);
	imshow("img",img8b);
	u.susanPrincipal(img8b,15);
	imshow("edges",img8b);
	waitKey(0);
}


void testVecDiff()
{
	vector<vector<ushort> > v1;
	vector<vector<ushort> > v2;
	v1.push_back((vector<ushort>){1,1,1});
	v1.push_back((vector<ushort>){2,2,2});
	v1.push_back((vector<ushort>){8,8,8});
	v2.push_back((vector<ushort>){3,3,3});
	v2.push_back((vector<ushort>){6,6,6});
	v2.push_back((vector<ushort>){7,7,7});
	UtilCpp u;
//	vector<ushort> r = u.findMaxDiff2(v1,v2);
//	for(uint i=0; i<r.size(); i++)
//		cout << r[i] << endl;

	//int featureValue=u.getMaxDiffMedian(v1,v2);

}

void testKeyPointFilter()
{
	vector<KeyPoint> v;
	v.push_back(KeyPoint(0,0,5));
	v.push_back(KeyPoint(1,1,5));
	v.push_back(KeyPoint(1,0,5));
	v.push_back(KeyPoint(0,1,5));
	cout << v.size() << endl;
	KeyPointsFilter::runByImageBorder(v,Size(3,3),1);
	cout << v.size() << endl;
}

void testCompareHistogramPETS()
{
	UtilCpp u;
	HessParticle p;
	Rect world(0,0,146,116);
	Mat img1 = imread("/home/alex/Desktop/alp/doutorado/06.Tese/01.Monografia/figuras/histogram/bgbox.png");
	Mat img2 = imread("/home/alex/Desktop/alp/doutorado/06.Tese/01.Monografia/figuras/histogram/bg.png");
	Mat mask = imread("/home/alex/Desktop/alp/doutorado/06.Tese/01.Monografia/figuras/histogram/mask.png");
	p.setWorld(world);

	Mat hsv1 = u.bgr2hsv(img1);
	Mat hsv2 = u.bgr2hsv(img2);
	float v1 = p.compareIntExtHistograms(hsv1, mask);
	float v2 = p.compareIntExtHistograms(hsv2, mask);
	cout << "sem o bilateral filter." << endl;
	cout << "v1=" << v1 << ", v2=" << v2 << endl;

	Mat bf1,bf2;
	bilateralFilter(img1, bf1, 13, 50, 20);//25, 50, 15
	bilateralFilter(img2, bf2, 13, 50, 20);//25, 50, 15
	Mat hsv1f = u.bgr2hsv(bf1);
	Mat hsv2f = u.bgr2hsv(bf2);
	float v1f = p.compareIntExtHistograms(hsv1f, mask);
	float v2f = p.compareIntExtHistograms(hsv2f, mask);
	cout << "com o bilateral filter." << endl;
	cout << "v1=" << v1f << ", v2=" << v2f;
}

void testCompareHistogramFlag()
{
	UtilCpp u;
	HessParticle p;
	Rect world(0,0,146,116);
	Mat img1 = imread("/home/alex/Desktop/alp/doutorado/06.Tese/01.Monografia/figuras/histogram/flag.png");
	Mat mask = imread("/home/alex/Desktop/alp/doutorado/06.Tese/01.Monografia/figuras/histogram/maskFlag.png");
	p.setWorld(world);

	Mat hsv1 = u.bgr2hsv(img1);
	float v1 = p.compareIntExtHistograms(hsv1, mask);
	cout << "v1=" << v1 << endl;
}

void testSampleBorder()
{
	GhostDetector gd;
	UtilCpp u;
	Mat roiRGB = imread("/home/alex/Desktop/alp/doutorado/06.Tese/05.Apresentacao/figuras/Ub.png");
	Mat mask = imread("/home/alex/Desktop/alp/doutorado/06.Tese/05.Apresentacao/figuras/Ub.png");
	Mat ero = imread("/home/alex/Desktop/alp/doutorado/06.Tese/05.Apresentacao/figuras/ero.png");
	cvtColor(mask,mask,CV_BGR2GRAY);

	cvtColor(ero,ero,CV_BGR2GRAY);

			imshow("Source",mask);
//			imshow("Selected Edges",mask);
//			imshow("Neighborhood",coroa);
			imshow("ero",ero);
			int count = cv::countNonZero(mask);
	//		imwrite("/media/alex/3afb1861-a12a-472d-9dda-ac2b0bdfb6c6/write_test/cdw/ghost/sum8b.bmp",sum8b);
	//		imwrite("/media/alex/3afb1861-a12a-472d-9dda-ac2b0bdfb6c6/write_test/cdw/ghost/sumth.bmp",sumth);
	//		imshow("sClone",sClone);
			float result;
			result = gd.isGhostBySamples(mask,ero,count);
			waitKey(0);
}
void testGetComplement()
{
	GhostDetector gd;
	vector<int> vAreas;
	vector<Point2f> cofm;
	vector<vector<Point> > meaningfulContours;
	vector<Rect> regions;
	Mat img = imread("testImages/test8.bmp");
	Mat img8b;
	cvtColor(img,img8b,CV_BGR2GRAY);
	gd.setMinArea(3);
//	vector<vector<Point> > contoursHull = gd.getContours(img8b, regions, cofm, meaningfulContours, vAreas);

	UtilCpp u;
	vector<Point> v=meaningfulContours[0];
	vector<KeyPoint> vkp = u.vecPoint2Keypoint(v);
	cout << vkp.size() << endl;
	KeyPointsFilter::runByImageBorder(vkp,img8b.size(),2);
	cout << vkp.size() << endl;
	vector<Point> vpNotAtBorder = u.vecKeyPoint2Point(vkp);
	vector<Point> r = u.getComplementSet(v,vpNotAtBorder);

//	v.push_back(Point(0,0));
//	v.push_back(Point(1,1));
//	v.push_back(Point(1,0));
//	v.push_back(Point(0,1));
//	vector<Point> v2;
//	v2.push_back(Point(0,0));
//	v2.push_back(Point(1,1));
	vector<vector<Point> > vcs;
	vector<vector<Point> > vcs2;
	vector<vector<Point> > vcs3;

	vcs.push_back(r);
	vcs2.push_back(vpNotAtBorder);
	vcs3.push_back(v);

	Mat sClone=Mat::zeros(img8b.size(),CV_8U);
	Mat sClone2=Mat::zeros(img8b.size(),CV_8U);
	Mat sClone3=Mat::zeros(img8b.size(),CV_8U);

	cv::polylines(sClone,vcs,0,Scalar(255));
	cv::polylines(sClone2,vcs2,0,Scalar(255));
	cv::drawContours(sClone3,vcs3,0,Scalar(255));

	imshow("img",img8b);
	imshow("sClone",sClone);
	imshow("sClone2",sClone2);
	imshow("sClone3",sClone3);

	waitKey(0);
}

int main(int argc, const char** argv)
{
	if(argc>1)
	{
		const string videoPath = argv[1];
		const string maskPath = argv[2];
		const string outPath = argv[3];
		const string inputPrefix = argv[4];
		const string extInput = argv[5];
		int firstfile=-1;
		int lastfile=0;
		int dataset=1;
		int step=1;
		if(argc>=7)
			firstfile=atoi(argv[6]);
		if(argc>=8)
			lastfile=atoi(argv[7]);
		if(argc>=9)
			step=atoi(argv[8]);

//		testCompareHistogramFlag();
//		testBS(videoPath, maskPath, inputPrefix, extInput, firstfile, lastfile, step);
//		getFeatures(videoPath, maskPath, outPath, inputPrefix, extInput, firstfile, lastfile, step);
//		testHessTracker(videoPath, maskPath, outPath, inputPrefix, extInput, firstfile, lastfile, step);
		testSampleBorder();
	}
}
