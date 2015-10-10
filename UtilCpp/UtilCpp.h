#ifndef UUTILCCP_H_
#define UUTILCCP_H_

#include <cstring>
//#include <string>
#include <fstream>
#include <ios>
#include <iostream>
#include <iomanip>
#include <vector>
#include "OOTestCase.h"
#include "Enums.h"
#include "csv_v3.h"
#include <gsl/gsl_qrng.h>

//#include "ULBSobel.h"

using namespace std;
using namespace cv;


#define SWAP(a,b) { \
        int c = (a);    \
        (a) = (b);      \
        (b) = c;        \
    }


typedef struct pixel_
{
	unsigned char r_h;
	unsigned char g_s;
	unsigned char b_v;
	int i;
	int j;
} pixel;

typedef struct pixel32f_
{
	Vec3f color;
	int i;
	int j;
} pixel32f;


typedef struct result_
{
	int fp;
	int fn;
	int tp;
	int tn;
} cresult;

typedef struct square_
{
	unsigned short x1;
	unsigned short y1;
	unsigned short x2;
	unsigned short y2;
} square;

typedef struct point_
{
	int x;
	int y;
}point;

typedef struct size_
{
	unsigned short height;
	unsigned short width;
}size;

typedef struct ellipse_
{
	Point center;
	Size axis;
	double phi;
}ooellipse;

typedef struct testcase_
{
int w;
int h;
int wh;
int bpp;
int firstfile;
int fnlength;
int nfiles;
int lastfile;
char bmphder[30];
char testname[30];
char seqname[30];
char testname_all[30];
char testname_sdelta[30];
char path[100];
char gtpath[100];
unsigned short GT[20];
cresult result;
} testdata;

typedef struct testcase3_
{
int w;
int h;
int wh;
int bpp;
string pathname1;
string pathname2;
unsigned char rgb1[640][400][3], rgb2[640][400][3];
unsigned char gt[640][400][3];
unsigned char bg[640][400][3];
unsigned char output[640][400][3];
int firstfile;
int fnlength;
int nfiles;
int ngts;
string testname;
string seqname;
string testname_all;
string testname_sdelta;
string path;
string gtpath;
string gtspath;
vector<string> * gtnames;
} testcase3;

typedef struct {
	unsigned char R;
	unsigned char G;
	unsigned char B;
} RGB;

//static RGB bbcolors[12] = {
//{ 255, 50, 0 },
//{ 50, 0, 255 },
//{ 0, 255, 50 },
//{ 50, 255, 0 },
//{ 255, 0, 50 },
//{ 0, 50, 255 },
//{ 150, 100, 30},
//{ 30, 100, 150 },
//{ 30, 150, 100 },
//{ 100, 150, 30 },
//{ 150, 30, 100 },
//{ 100, 30, 150 }
//};

struct BMP
{
  //char Type[2];           //File type. Set to "BM".
  unsigned long Size;     //Size in BYTES of the file.
  unsigned long Reserved;      //Reserved. Set to zero.
  unsigned long OffSet;   //Offset to the data.
  unsigned long headsize; //Size of rest of header. Set to 40.
  unsigned long Width;     //Width of bitmap in pixels.
  unsigned long Height;     //  Height of bitmap in pixels.
  unsigned short  Planes;    //Number of Planes. Set to 1.
  unsigned short  BitsPerPixel;       //Number of Bits per pixels.
  unsigned long Compression;   //Compression. Usually set to 0.
  unsigned long SizeImage;  //Size in bytes of the bitmap.
  unsigned long XPixelsPreMeter;     //Horizontal pixels per meter.
  unsigned long YPixelsPreMeter;     //Vertical pixels per meter.
  unsigned long ColorsUsed;   //Number of colors used.
  unsigned long ColorsImportant;  //Number of "important" colors.
}; typedef struct BMP header;


namespace Util {

#define CLASS_WP_FA_0 1
#define CLASS_WP_FA_255 10
//#define CLASS_FG 255
#define CLASS_BG 0
#define FALSE_P	20
#define FALSE_N	40
#define FALSE_NS	60
#define FALSE_NFG 80
#define FALSE_PFG	130
#define FALSE_PS	140
#define TRUE_PS	100
#define TRUE_PFG	120
#define THRS_WP 0.8
#define byte unsigned char

//#define DEBUG

//#define PRINT_DEBUG

//#define COMPARE_PIXEL_WISE
//#define DEBUG_RESULT
//#define COUNT_MAX_SDELTA
#define SMALL_VSSN06
//#define PERFORMANCE_EVALUATION
//#define EXTENDED
#define DELAY_ON
//#define EXTENDED2

#define IR			1
#define HW1			2
#define HW2			4
#define HW3			8
#define CAMPUSSH	16
#define LABORATORY	32

#define PRATI2003 			1
#define PRATI2003ALEX 		2
#define ELGAMMAL2003 		4
#define ELGAMMAL2003ALEX 	8

#define CHANGECOLOR			1
#define ERRORS_SUM	 		2
#define SIMILARITY	 		4
#define COMPARE_ONE		 	8
#define WRITE_RESULT		16
#define COMPARE_20 			32
#define SUM_DIFF 			64
#define COMPARE_ALL			128
#define ALLPICTURES			256
#define WRITE_BACKGROUND	512

#define H 300
#define W 300
#ifndef MAX
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))
#endif
#ifndef ABS
#define ABS(a)	   (((a) < 0) ? -(a) : (a))
#endif



class UtilCpp
{
public:
void printStatistcs(int fit, int wh, string & str);
unsigned char readRGB(const char * path, unsigned char * img, int w, int h, int bpp);
unsigned char readRGB(string & path, unsigned char * img, int w, int h, int bpp);
char readRGB_OpenCV(const char * path, unsigned char * img);
unsigned char readYCBCR444( char * path, unsigned char * img, int w, int h);
void writeRGB(string & path, unsigned char *img, int w, int h, int bpp);
void writeRGB(string name, Mat & img, int n=-1);
void writeJPG(string name, Mat & img, int n=-1);
void writePNG(string name, Mat & img, int n=-1);
void writeImg(string name, Mat & img, int n, string ext);
void writeRGB_OpenCv(string & path, unsigned char *img, int w, int h, int bpp);
void writeRGBWithHeader(string & path, string & hdr, unsigned char *img, int w, int h, int bpp);
void writeRGBWithHeader(char * path, const char * headerpath, unsigned char *img, int w, int h, int bpp);
void writeBackground(int nfile, Mat & img, int w, int h);
void writeBackground(int nfile, Mat & img);
void writeBackground_OpenCv(int nfile, unsigned char * bg, int w, int h);
void writeTest_OpenCv(int nfile, unsigned char * bg, int w, int h);
void writeTest_OpenCv(int nfile, IplImage *image, int w, int h);
void writeResult(int nfile, unsigned char * img, const string * hdrpath, int w, int h, int bpp);
void writeResult(int nfile, unsigned char * img, int w, int h);
void writeResult(int nfile, Mat &img);
void writeResult(int nfile, Mat &img, string spath);
void RGBtoBWImage(unsigned char *p, unsigned char *bw, int bwlen);
int BWtoRGBImage(unsigned char *bw, unsigned char *rgb, int w, int h);
void RGBtoBWImage1(unsigned char *p, unsigned char *bw, int bwlen);
void bmpHeader(ofstream & ofile, int h, int w, int bps);
void RGBtoHSV( unsigned char cr, unsigned char cg, unsigned char cb, float *h, float *s, float *v );
void YUVtoRGB(unsigned char y, unsigned char cb, unsigned char cr, unsigned char *r, unsigned char *g, unsigned char *b );
void countErrors(Mat & gt, Mat & pic, cresult & r);
//Vec4f countErrors(Mat & gt, Mat & pic);
void countErrorsShadowForeground(unsigned char *gt, unsigned char * imresult, float *ni, float *e, int w, int h, int bpp);
void changeColor(byte * in, byte *out, int c, int th, int len, bool eq);

int comparePixel(unsigned char gt, unsigned char ft);
int comparePixelShadowForeground(byte ir, byte ig, byte ib, byte bgr, byte bgg, byte bgb);
void connectedComponents(unsigned char * img, int w, int h, unsigned char thr, unsigned char * labels, float wpthrs, int jointDifference);
float probVerifyConnectivity8(unsigned char * p, int w);
float probVerifyConnectivity16(unsigned char * p, int w, float *prob8);
float probVerifyConnectivityAfter(unsigned char * p, int w);
void setAfterProb(float p5, float p6, float p7, float p8);
short verifyConnectivity(unsigned char * p, int w);
float verifyColor(unsigned char *p, int w, unsigned char wcolor);
short verifySimilarFGColor(unsigned char *p, int w);
short verifySimilarFGColor(Mat & img, int y, int x);
short verifyConnectivity(Mat & src, int y, int x);
int verifyConnectivityAfter(unsigned char * p, int w);
void setProb(float p0, float p1, float p2, float p3, float p4);
void setProb16(float p0, float p1, float p2, float p3, float p4, float p5, float p6, float p7, float p8, float p9, float p10, float p11);
int compareImages(Mat & img1, Mat & img2, int w, int h, bool print);
void writeDifference(unsigned char * im, unsigned char *gt, unsigned short h, unsigned short w, int nfile);
char * createFilename(char * path, char * filename, int number, int characters, string & ext);
string createFilename(string & path, int number, int characters, string & ext);
void reportArray(unsigned short * array, int l, int c, char * filename);
float evPoly(float * coefs, unsigned char order, int x);
void setIterfiles(int iterfiles) { this->iterfiles = iterfiles; }
int getIterfiles() const { return iterfiles; }
void drawSquare(unsigned char * img, unsigned short w, square *sqr, unsigned char v);
void drawRGBSquare(unsigned char * img, unsigned short w,  unsigned short h, square *sqr, unsigned short objID, bool writeNumber);
bool compareSquares(square *s1, square *s2);
bool isInside(square *sqr, unsigned short x, unsigned short y);
vector<pixel> bresenhamline(int x0, int y0, int x1, int y1, int max);
vector<Point> getOuterContour(Mat & src, int x0, int y0, int x1, int y1);
int countOnLine(Mat & src, int x0, int y0, int x1, int y1, int checkValue);
void capHoles(Mat & src);
void convert1to255(unsigned char *src, unsigned char *dest, int len);
Mat sideBySideImgs(Size imgsize, Mat & m1, Mat & m2, Mat & m3, Mat & m4, bool resizing);
void writeToLogFile(int n, bool endline);
Mat getFrame(bool hsv, string path, int currfile, int fnlength, string ext);
Mat getFrame(bool hsv, int currfile, OOTestCase * tc);
Mat convertTo8b(Mat & src);
Mat convertToPositive(Mat & src);
void videoSetup(Mat firstImg, string file);
void addFrameToVideo(Mat img);
void finishVideoWriting();
void drawText(char * txt, IplImage* image, cv::Point *orig);
void drawText(Mat & src, int type, string text, float number=-1000000, Point orig=Point());
Mat getRaimbowGradient(Mat hsv, Mat gray);
Mat getRGBHistogram( Mat src, vector<float> &hd, bool render );
Mat convertTo3Channels(Mat & src);
Mat getOneChHistogram( Mat src, float tc, float MAD, int type, vector<float> &hd, bool render );
int getLinePoint(float a, float b, int x);
float calcMSE(vector<Point2f> v);
Mat removeNoise(Mat src);
void removeNearbyNoise(Mat & src, bool preserveOrig, int cmin=15, int sumth=2);
void save(float * src, int n, string & path);
void applySobel(Mat & src);
void applyScharr(Mat & src);
Mat highPass(Mat & src);
float * load(int n, string & path);
bool compareArray(float *src1, float *src2, int n);
bool rectAtBorder(Rect& r, Rect & world, int minDiagDistance);
Mat getLargestContour(Mat & mask, Rect & largestRect, vector<Point> & lvp, int * count=0);
Mat bgr2hsv(Mat input);
Mat hsv2bgr(Mat input);
string floatToString(vector<float> vf, int precision=3);
string floatToString(float f);
string Vec3fToString(Vec3f f);
string intToString(int n);
string boolToString(bool b);
void vectorToCSV(string file, vector<float> v);
void printVector( vector<int> & v);
void printVector( vector<float> & v, bool withcomma=false);
void printVector( vector<string> & v);
void printArray( float * v, int n);
void appendToFile(map<string,string> vs, string path);
void appendToFile(vector<string> & vs, string path, bool addComma=false, uint ncache=1);
void appendToFile(string s, string path);
bool compute_IH3C( Mat & I , vector<vector < Mat > *> * vec_II, int nbins );
bool compute_IH( Mat & I , vector < Mat >* vec_II, int nbins );
vector < Mat > getIH(Mat & I, int nbins);
bool compute_histogram ( int tl_y , int tl_x , int br_y , int br_x , vector < Mat >* iiv , vector < float >& hist );
void Get_Pixel_Bin ( Mat & I , Mat & bin_mat, int nbins);
int histo_bin( float h, float s, float v );
bool fileExists(string name);
Mat vecToMat(vector<float> v);
float weightedMedian(vector<float> x, vector<float> w);
float median(vector<float> vec);
ushort median(vector<ushort> vec);
short median(vector<short> vec);
uchar median(vector<uchar> vec);
float medianOrLower(vector<float> vec);
float getMean(vector<float> v);
int getMean(vector<ushort> v);
float stdDev(vector<float> v);
Vec3f fitLine(double *x, double *y, int n);
Mat equalizeIntensity(const Mat& inputImage);
Mat getLRG(Mat & src);
Mat lsbpEdges(Mat & src, int th=20, int type=0);
vector<vector<ushort> > getLBSP(vector<KeyPoint> vp, Mat & src, const uchar * ref=NULL);
vector<vector<ushort> > getLbspSet(vector<KeyPoint> vp, vector<KeyPoint> vref, Mat & src);
vector<KeyPoint> getRandomKeyPoints(int w, int h, int grid);
vector<Point> getRandomPoints(int w, int h, int grid);
float evalOutContourSamples(Mat & src, Mat & ero, vector<Point> & vp, vector<Point> & selectedp, int angle_step=30, Mat out=Mat());
void randomPatches(int height, int width, float n, vector<Point> contour, vector<Rect>& patch_vec);
int evalOutContour(Mat & src, Mat & out, int angle_step, Point p, vector<Point> & selectedp);
vector<Point> sampleBorder(Mat borderMask, Mat ero);
void fixedPatches(int height, int width, Size grid, vector<Rect>& patch_vec);
void fixedPatches(int height, int width, int n, vector<Rect>& patch_vec, int type = RECT_PATCHES);
void drawPatches(Mat & src , vector<Rect> & patches, Scalar color);
void plotLineAndPoints(Mat & src, Vec3f line, vector<double> & x, vector<double> & y);
uint getHammingDiff(vector<ushort> v1, vector<ushort> v2);
vector<ushort> findMinDiff2(vector<vector<ushort> > array1, vector<vector<ushort> > array2);
vector<ushort> findMaxDiff(vector<vector<ushort> > array1, vector<vector<ushort> > array2);
vector<ushort> findMinDiff(vector<vector<ushort> > array1, vector<vector<ushort> > array2);
uint getMaxDiffMedian(vector<vector<ushort> > array1, vector<vector<ushort> > array2);
void drawPoints(vector<Point> v, Mat & src, Scalar color);
void drawKeyPoints(vector<KeyPoint> v, Mat & src, Scalar color);
vector<KeyPoint> vecPoint2Keypoint(vector<Point> v, int kpsize=1);
vector<Point> vecKeyPoint2Point(vector<KeyPoint> v);
vector<Point> getComplementSet(vector<Point> v1, vector<Point> v2);
vector<float> subtract(vector<float> & data1, vector<float> & data2);
Mat susansmooth(Mat & src, double t, double sigma);
IplImage* susanSmooth(IplImage *src, int w, int h, double t, double sigma);
double calcSusanSmooth(IplImage* src, int x, int y, double t,double sigma,int w, int h);
void susanPrincipal(Mat & src, int t);
void susanEdges3C(Mat & src, int t);
void susanEdges(Mat & src, int t);
IplImage* wrapCreateImage32F(const int width, const int height, const int channels);
void zeroNegatives(vector<float> & v);
vector<uchar> getMedian3ch(vector<vector<short> > v);
void eraseByValue(std::vector<short> & myNumbers_in, int number_in);
Mat proportionalMedianBlur(Mat & src);
vector<Size> getLargerGrid(Size g, Rect r, int scale);
vector < Rect > findBestPatches(vector<Size> psize, Mat& r1, Mat& r2, float disrate, vector<float> weights);
std::vector<float> apply_permutation( std::vector<float> & vec, std::vector<int> & p);
std::vector<int> sort_permutation(vector<float> const & w);
std::vector<int> apply_permutation( std::vector<int> & vec, std::vector<int> & p);
std::vector<Size> apply_permutation( std::vector<Size> & vec, std::vector<int> & p);
std::vector<int> sort_permutation(vector<int> const & w);
void sortTwoVectors(vector<int> & src1, vector<int> & src2);
inline bool exists_test(const std::string& name);
void removedupes(std::vector<Point2f> & vec);
vector<vector<int> > getFeatureList(string csvfile);
private:
struct sortstruct
{
	// sortstruct needs to know its containing object
	//UtilCpp * m;
	sortstruct() {};

	// this is our sort function, which makes use
	// of some non-static data (sortascending)
	bool operator() ( Point i, Point j )
	{
//			if ( m->sortascending )
//				return i < j;
//			else return i > j;
		float modi = sqrt(i.x*i.x+i.y*i.y);
		float modj = sqrt(j.x*j.x+j.y*j.y);
		return (modi<modj);
	}
};

void sortPointVec  (vector<Point> vec)
{
	// create a sortstruct and pass it to std::sort
	sortstruct s;//(this);
	std::sort ( vec.begin (), vec.end (), s );
//	for (uint i = 0; i < vec.size(); i++)
//		std::cout << vec[i].x << "," << vec[i].y << std::endl;
}



VideoWriter * videoOutput;
static float prob12[12];
int iterfiles;
unsigned char irgb[H][W][3];
};
}


#endif /*UTILCCP_H_*/
