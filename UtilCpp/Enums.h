/*
 * Enums.h
 *
 *  Created on: Jun 20, 2010
 *      Author: alex
 */

#ifndef ENUMS_H_
#define ENUMS_H_
#include <opencv2/opencv.hpp>
#include "OOTestCase.h"
#include <list>
#include <string>
#define EXTENDED
#define CSUM_LEN 800
#define MIM_SIZE 4
#define SHOWIMG
#define SHOW_ALL 1
#define SHOW_SELECTED 2
using namespace std;
using namespace cv;
//using namespace Util;
enum subresult {SUB_INTER2=200, SUB_INTER3=254, SUB_POSITIVE=100, SUB_NEGATIVE=50, SUB_OUT=0};
enum ftnumbers { AVER_BACKGROUND_MANHATAN=1, PRATI_SHADOW_FILTER=2, PBASORIG_CLSF=4,  FRAME_DIFF_MANHATAN=8,  SJDC=16, MORPHOLOGIC_FILTER=32, HOFMANN_CLSF=64, PBAS_HSV_CLSF=128, PBAS_RGB_CLSF=256, KDEHSV_CLSF=512, ROSIN_RGB_CLSF=1024, ROSIN_HSV_CLSF=2048, PARK2010_CLSF=4096};
enum param_type { TSU, TSB, TI, TB, ALPHA, W1, W2, WB, TAUHS, TAUH, TAUS, TAUV1, TAUV2, TCR, TVC, ELGTHC, ELGK, LNTHI, LNTHE, SUMTH, CFA, CFB, CFC, CFD, CSUM_LENGTH, NCOEFFS, PROB_ARRAY};
enum classes { FG=255, BG=0};
enum feturetype { BACKGROUND_SUB, SHADOW_FILTER, FRAME_DIFF, CORRECTION, SUM, FRAME_WISE };
enum gdatatype { CHAR, SHORT, INT, FLOAT };
enum outputtype {WR_RESULT=1, WR_BACKGROUND=2, PXWISE=4, CALC_SIMILARITY=8, WR_ALL=16, ERRSUM=32, FASTEVAL=64, COUNTSUM=128, IMSHOW=256, CALC_METRICS=512};
enum metrictypes {MANHATAN, TRUNCATED_DIFF};
enum backgroundtypes {AVER_BACKGROUND, IIR_PARK_2010};
enum genericmethods {CORRECT_BGND_MODEL, SET_DIFF_DELAY, DIFF_ONE_FRAME, UPDATEPW, RESETPW, UPDATETC};
enum test_set {CDW_SET=1, AVSS_SET=2,INESC=3, WALLFLOWER=4, IRT=5};
enum test_cases {WCA, WMO, WTOD, WFA, WLS, WWT, WBO, ICAMPUS, IMEETINGROOM, IWATERSURFACE, IFOUNTAIN, ISC, IAIRPORT, ILOBBY, IBOOTSTRAP, ISS, AIR, ACAMPUS, LAB, SIMPLETEST, CDW_HIGHWAY, CDW_OFFICE, CDW_PEDESTRIANS, CDW_TPETS2006, LIVE_CAM};
enum operations {SUMM,SUB,MULT,DIV};
enum featanalysertype {ADHOC_FEATURE_ANALYSER=1, DD_FEATURE_ANALYSER=2, NO_FEATURE_ANALYSER=4};
enum ellipse_idcode{AXESCOLOR=1,AXESRATE=2};
enum votes{MIN_VOTE=-16, MIDMIN_VOTE=-8, UNCERTAIN_VOTE=0, MIDMAX_VOTE=8, MAX_VOTE=16};
enum colorchannel{CH_HSV=1, CH_RGB=2, CH_HS=3, CH_V=4, CH_HSV32F=5};
enum model_appearence{SUPERPIXELS=1, HISTOG_PATCHES=2, HISTOGRAM=3};
enum creation_reason{INITIAL_CREATION=1, REASON_2=2, REASON_3=3, REASON_4=4};
enum patch_types{SQUARE_PATCHES, RECT_PATCHES};
enum obj_type{GHOST,VALID_OBJ,THIN_AREA};
enum text_type{TITLE,SUB_TITLE,SECTION,SUB_SECTION};
enum edge_type{SUSAN_EDGES,GPB_EDGES};
typedef struct genericdata_
{
	double d1;
	double d2;
	unsigned char d3;
	short d4;
	string type;
	short *pd;
	double *d5;
	bool valid;
	Vec4i dv1;
	Vec2f dv2;
	Vec3b dv3;
	short vote;
} genindata;

namespace Util
{
class data
{
public:
	Mat currpic;
	Mat previouspic;
	Mat currpichsv;
	Mat currpichsv32f;
	Mat currout;
	Mat lastout;
	Mat bgImg;
	Mat bgImgHsv32f;
	Mat gtImg;
	Mat trackImg;
	Mat ellipseImg;
	Mat segmented;
	Mat DebugResult;
	Mat DebugResult2;
	Mat rendVotes;
	Mat bluredImage;
	Mat bluredImageHsv;
	Mat bluredImageHsv32f;

	vector<Mat> ppicblur;
	vector<Mat> ppichsvblur;
	vector<Mat> ppic;
	vector<Mat> ppichsv;
	vector<Mat> ppichsv32f;
	vector<Mat> output;
	vector<Mat> trainImagesRGB;
	vector<Mat> trainImagesHSV;
	vector<Mat> trainImagesHSV32F;
	deque<Mat> grabOutput;


	unsigned char curroutn; //0 a 4
	//	unsigned short curroutfilen;
	//	unsigned char * bg;
	int w;
	int h;
	int wh;
	int config;
	int outcfg;
	int piclen;
	int outlen;
	int currn;
	bool cmp;
	bool tracked;
	//	char r;
	//	char g;
	//	char b;
	//	long currcsum[CSUM_LEN][CSUM_LEN];
	//	long totcsum[CSUM_LEN];
	//	unsigned short epsum[2000];
	//	unsigned short ensum[2000];
	int currfile;
	int lastfile;
	OOTestCase *tc;
	int nOfTrainningFiles;
};
};
//class traindata
//{
//public:
//
//}

#endif /* ENUMS_H_ */
