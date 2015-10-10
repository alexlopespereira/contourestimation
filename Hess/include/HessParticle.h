/*
  Perform single object tracking with particle filtering

  @author Rob Hess
  @version 1.0.0-20060306
*/

#ifndef HESSPATCHES_H_
#define HESSPATCHES_H_
#include "opencv2/opencv.hpp"
#include "particleDefs.h"
#include "observation.h"
#include "Enums.h"
#include "HessParticle.h"
#include "UtilCpp.h"
/* From GSL */
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <numeric>
#include <map>

using namespace cv;
using namespace Util;
using namespace std;


/******************************** Definitions ********************************/
#define DEBUG
//#define PETS
//#define AVSS
//#define CDW
#define MOMENTS
//#define GT
//#define WRITE_GHOSTS
//#define WRITE_GHOSTS_TRACKING
#define EVAL_DISPLACEMENT
#define USE_PATCHES
//#define CREATE_EDGES_STDDEV
//#define CREATE_HEADER
//#define ALL_FEATURES
#define MULTIPLE_FRAME_DETECTION
//#define BATTACHARYYA
//#define TRAINING

//#define F7
#define F5
#define F4
#define F3
#define TRACKING
//#define EVAL_SVM
//#define EVAL_NAIVE_BAYES
//#define CVUI
/* maximum number of frames for exporting */
#define MAX_FRAMES 2048
#define TESTING
//#define COUNT_CONTOURS
#define SUSAN
#define ASOD
//#define DENOISE
//#define OUTPUTALL
//#define REPORT_RESULT

/********************************* Structures ********************************/

/* maximum number of objects to be tracked */
#define MAX_OBJECTS 2

/* standard deviations for gaussian sampling in transition model */
#define TRANS_X_STD 0.7
#define TRANS_Y_STD 0.7
#define TRANS_S_STD 0.01

/* autoregressive dynamics parameters for transition model */
#define A1  2.0
#define A2 -1.0
#define B0  1.0000

#define PI 3.14159265

static int particle_cmpHT( const void* p1, const void* p2 )
{
	particle* _p1 = (particle*)p1;
	particle* _p2 = (particle*)p2;

	if( _p1->w > _p2->w )
		return -1;
	if( _p1->w < _p2->w )
		return 1;
	return 0;
}

namespace HessTracking {

class HessParticle {

public:
	HessParticle(){
		minArea=200;
		objMinArea=10000000;
		objMaxArea=0;
		objOldArea=0;
		countAreaChanges=0;
		acumAreaChanges=0;
		analyzedCount=0;
		stddev=0;
		displTries=0;
		areaTries=0;
		reason=-1;
		incw=inch=incx=incy=0;
		minDiagDistance=0;
		nrows_cols=0;
		perwdiag=0;
//		f3Average=0.4;
//		f4Average=0.4;
		alphaF=0.75;
		//age=0;
		//validCount=0;
		//acumWidth=0;
		//orig_h=0;
		//orig_w=0;
		//isLineFitted=false;
	}

	~HessParticle(void)
	{
		//free(tparticles);
	}

	void printArray( float *v, int n)
	{
		printf("\n");
		for(int i=0; i<n; i++)
		{
			printf("%f ", *v++);
		}
		fflush(stdout);
	}

	void printVector( vector<int> & v)
	{
		printf("\n");
		for(uint i=0; i<v.size(); i++)
		{
			printf("%d ", v[i]);
		}
		fflush(stdout);
	}

	void printVector( vector<float> & v)
	{
		printf("\n");
		for(uint i=0; i<v.size(); i++)
		{
			printf("%f ", v[i]);
		}
		fflush(stdout);
	}

	void printParticles( particle * p, int n)
	{
		for(int i=0; i<n; i++)
		{
			cout << "Rect(" << p->r.x << "," << p->r.y  << ")" << "[" << p->r.width << "," << p->r.height <<"]"<< "=" << p->w << endl;
			p++;
		}
		cout.flush();
	}

	int getNormalizedHistogram(float* h, int n, Mat & img, Mat mask = Mat())
	{
		int count = calc_histogramMat(h, n, img, mask);
		normalize_vhist(h, n);
		return count;
	}

	int getNormalizedHistogram(float vh[][NH*NS + NV], int n, vector<Rect> & patchs, Mat & img, Mat mask = Mat())
	{
		int count=0;
		for(uint i=0; i< patchs.size(); i++)
		{
			Mat pRoiF(img,patchs[i]);
			Mat pRoiM;
			if(mask.data)
				pRoiM=Mat(mask,patchs[i]);
			count+=getNormalizedHistogram(&vh[i][0], n, pRoiF, pRoiM);
		}
		return count;
	}

	int histo_bin( float h, float s, float v )
	{
	  int hd, sd, vd;

	  /* if S or V is less than its threshold, return a "colorless" bin */
	  vd = MIN( (int)(v * NV / V_MAX), NV-1 );
	  if( s < S_THRESH  ||  v < V_THRESH )
	    return NH * NS + vd;

	  /* otherwise determine "colorful" bin */
	  hd = MIN( (int)(h * NH / H_MAX), NH-1 );
	  sd = MIN( (int)(s * NS / S_MAX), NS-1 );
	  return sd * NH + hd;
	}

	int calc_histogramMat(float * histo, int n, Mat & img, Mat & mask )
	{
		//float* histo;
		Mat h,  s,  v;
		//float* hist;
		int r, c, bin;
		vector<Mat> planes;
		memset(&histo[0], 0, n*4);
		/* extract individual HSV planes from image */
		split(img, planes);
		h=planes[0];
		s=planes[1];
		v=planes[2];
		/* increment appropriate histogram bin for each pixel */
		int count=0;
		if(!mask.data)
		{
			for( r = 0; r < img.rows; r++ )
				for( c = 0; c < img.cols; c++ )
				{
					bin = histo_bin( h.at<float>(r, c ),s.at<float>(r, c ),v.at<float>(r, c ));
					histo[bin] += 1;
				}
			count=img.rows*img.cols;
		}
		else
		{
//			bool visited=false;
			for( r = 0; r < img.rows; r++ )
				for( c = 0; c < img.cols; c++ )
				{
					if(mask.at<uchar>(r,c)==255)
					{
						bin = histo_bin( h.at<float>(r, c ),s.at<float>(r, c ),v.at<float>(r, c ));
						histo[bin] += 1;
						count++;
//						visited=true;
					}
				}
//			if(visited==false)
//				cerr << "visited==false" << endl;
		}
		v.release();
		h.release();
		s.release();
		return count;
	}

	int check_hist( float * histo, int n )
	{
		float sum = 0;
		int i;
		/* compute sum of all bins and multiply each bin by the sum's inverse */
		for( i = 0; i < n; i++ )
		{
			sum += histo[i];
		}
//		if(sum>2)
//		{
//			cerr << "sum>2 = " << sum << endl;
//		}
		return sum;
	}
	void normalize_vhist( float * histo, int n )
	{
		float sum = 0, inv_sum;
		int i;
		/* compute sum of all bins and multiply each bin by the sum's inverse */
		for( i = 0; i < n; i++ )
		{
			sum += histo[i];
		}
		if(sum>0)
		{
			inv_sum = 1.0 / sum;
			for( i = 0; i < n; i++ )
				histo[i] *= inv_sum;
		}
//		else
//		{
//			cerr << "sum<=0" << endl;
//		}
	}

	void updateWeights(Mat & hsvf, Mat mask)
	{
		float s;
		bool allZero=true;
		updateParticlesHistogram(hsvf, mask);
		int countRate[np];
		float trustRates[np];
		vector<Rect> rects(np);
		int biggestCount=0;
		for( int j = 0; j < np; j++ )
		{
			s=tparticles[j].s;
			Rect rec(cvRound(tparticles[j].x) - cvRound(tparticles[j].width * s) / 2,
					cvRound(tparticles[j].y) - cvRound(tparticles[j].height * s) / 2,
					cvRound(tparticles[j].width * s), cvRound(tparticles[j].height * s));
			rec = rec & world;
			rects[j]=rec;
			if(!tparticles[j].toEval)
			{
				tparticles[j].w=0;
				countRate[j]=0;
				continue;
			}
			adjustParticle(hsvf,mask,&tparticles[j],rec);
			tparticles[j].w = likelihood(hsvf, mask, rec, tparticles[j].vhistos, nrows_cols, tparticles[j].n, &countRate[j]);

			if(countRate[j]>biggestCount)
				biggestCount=countRate[j];

			if (tparticles[j].w != 0)
				allZero = false;

			if (tparticles[j].w < 0)
				cerr << "tparticles[j].w= " << tparticles[j].w << endl;
		}
		//cout << "ID: " << objectID << endl;
		if(biggestCount>0)
		{
			for(int i=0; i<np; i++)
			{
				trustRates[i]=countRate[i]/(float)biggestCount;
				//cout << countRate[i] << ", " << trustRates[i] << ", " << tparticles[i].w << ", ";
				tparticles[i].w *= trustRates[i];
				//cout << tparticles[i].w << endl;
		#ifdef DEBUG
		//		cout << "count: " << count << endl;
//				Mat mroi = Mat(mask,rects[i]);
//				mroi = getLargestContour(mroi);
//				drawText(tparticles[i].w, 0.45, mroi, 140);
//				//imshow("imgroi",tmp);
//				imshow("mroi",mroi);
//				imshow("cmpmask",mask);
//				waitKey(1);
		#endif
			}
		}
		else
		{
			//cerr << "biggestCount<=0" << endl;
			weight += -1;
		}
		if (allZero)
			weight += -4;
	}

	void transition(int w, int h, gsl_rng* rng) {
		for (int j = 0; j < np; j++) {
			//int ch = tparticles[j].height;
			//int cw = tparticles[j].width;
			for(int i=0; i< nrows_cols*4; i++)
			{
				check_hist(&tparticles[j].vhistos[i][0],tparticles[j].n);
			}
			tparticles[j] = callTransition(tparticles[j], w, h, rng);
			//			if(tparticles[j].height!=ch)
			//				cout << ch << ", " << tparticles[j].height << endl;
			//			if(tparticles[j].width!=cw)
			//				cout << cw << ", " << tparticles[j].width << endl;

		}
	}

	void adjustParticle(Mat & hsv, Mat & mask, particle * p, Rect & newRec)
	{
		Mat mroi_before(mask,newRec);
		//imwrite("mroi_before.bmp",mroi_before);
		int countContours;
		Rect largestRect;
		vector<Point> vp;

		Mat mroi_result = getLargestContour(mroi_before, largestRect, vp, &countContours);
//		if(objectID==0)
//		{
//			imshow("roi_before",mroi_before);
//			if(mroi_result.cols>0)
//				imshow("mroi_result",mroi_result);
//			waitKey(0);
//		}
		if(largestRect.width*largestRect.height<minArea/2
				|| largestRect.width<nrows_cols || largestRect.height<nrows_cols
				|| largestRect.width<4 || largestRect.height<4)
		{
			p->toEval=false;
			return;
		}
		largestRect.x+=p->r.x;
		largestRect.y+=p->r.y;
//			if(countContours>1)
//			{
//				imshow("mroi_before",mroi_before);
//				imshow("mroi_result",mroi_result);
//				waitKey(0);
//		Mat roih(hsv,largestRect);
//		int width = largestRect.width;
//		int height = largestRect.height;
//		int x = largestRect.x + width / 2;
//		int y = largestRect.y + height / 2;

//		vector < Rect > patches;
//		fixedPatches(height, width, nrows_cols, patches);
//		p->x0 = p->xp = p->x = x;
//		p->y0 = p->yp = p->y = y;
//		p->width = width;
//		p->height = height;
//		p->r = largestRect;
//		getNormalizedHistogram(p->vhistos, p->n, patches, roih, mroi_result);
		newRec=largestRect;
	}

	void adjustParticles(Mat & hsv, Mat mask)
	{
		//int oldnrc=nrows_cols;
		int area = contourArea(contourReal); //tparticles[0].width*tparticles[0].height;
		nrows_cols = sqrt(area / minArea);
		if (nrows_cols > 8)
			nrows_cols = 8;

		if(nrows_cols<1)
			nrows_cols=1;
		for (int j = 0; j < np; j++)
		{
			if(tparticles[j].width*tparticles[j].height<minArea/2)
			{
				tparticles[j].toEval=false;
				continue;
			}
			Mat mroi_before(mask,tparticles[j].r);
			//imwrite("mroi_before.bmp",mroi_before);
			int countContours;
			Rect largestRect;
			vector<Point> vp;

			Mat mroi_result = getLargestContour(mroi_before, largestRect, vp, &countContours);
			if(objectID==0)
			{
				imshow("roi_before",mroi_before);
				if(mroi_result.cols>0)
					imshow("mroi_result",mroi_result);
				waitKey(0);
			}
			if(largestRect.width*largestRect.height<minArea/2
					|| largestRect.width<nrows_cols || largestRect.height<nrows_cols
					|| largestRect.width<4 || largestRect.height<4)
			{
				tparticles[j].toEval=false;
				continue;
			}
			largestRect.x+=tparticles[j].r.x;
			largestRect.y+=tparticles[j].r.y;
//			if(countContours>1)
//			{
//				imshow("mroi_before",mroi_before);
//				imshow("mroi_result",mroi_result);
//				waitKey(0);
				Mat roih(hsv,largestRect);
				int width = largestRect.width;
				int height = largestRect.height;
				int x = largestRect.x + width / 2;
				int y = largestRect.y + height / 2;

				vector < Rect > patches;
				u.fixedPatches(height, width, nrows_cols, patches);
				tparticles[j].x0 = tparticles[j].xp = tparticles[j].x = x;
				tparticles[j].y0 = tparticles[j].yp = tparticles[j].y = y;
				tparticles[j].width = width;
				tparticles[j].height = height;
				tparticles[j].r = largestRect;
				getNormalizedHistogram(tparticles[j].vhistos, tparticles[j].n, patches, roih, mroi_result);
				tparticles[j].w = 0;
//			}
		}
	}

	void initParticles(Mat& frameHSV, Mat& mask, Rect& region,
			vector<Point> contHull, vector<Point> contReal, Point2f& centermass, int p, int reason_) {
		//age=0;
		matchCount = 0;
		//validCount = 0;
		//unifNeighCount = 0;
		np = p;
		reason = reason_;
		incx = incy = incw = inch = 0.8;
		//orig_w=region.width;
		//orig_h=region.height;
		displaced = false;
		acumDisplacement = 0;
		//acumHeight=0;
		//acumWidth=0;
		//toRemove = 0;
		//countDisplaced = 0;
		//bigDisplacement = false;
		match = true;
		cOfMass = centermass;
		notGhost=0;
		//isLineFitted=false;
		int maxbins = NH * NS + NV;
		int area = region.width * region.height;
		int nbins = area;
		nbins = max(10, nbins);
		nbins = min(nbins, maxbins);
		float x, y;
		int width, height, k = 0;
		tparticles = (particle*) (((malloc(p * sizeof(particle)))));
		contourHull = contHull;
		contourReal = contReal;
		/* create particles at the centers of each of n regions */
		//  for( i = 0; i < n; i++ )
		//    {
		//		int imgdiag = sqrt(mask.cols*mask.cols+mask.rows*mask.rows);
		//		int minArea = std::pow(imgdiag*0.04,2);
		nrows_cols = sqrt(area / minArea);
		if (nrows_cols > 8)
			nrows_cols = 8;

		vector < Rect > patches;
		width = region.width;
		height = region.height;
		u.fixedPatches(height, width, nrows_cols, patches);

		x = region.x + width / 2;
		y = region.y + height / 2;
		objectDiagonal=sqrt(region.height * region.height + region.width * region.height);
		Mat roih(frameHSV, region);
		Mat roim(mask, region);
		setObservedRect(region);
		setLastObservedRect(region);
		cmtrackx.push_back(centermass.x);
		cmtracky.push_back(centermass.y);
		pastOrig = Vec2f(x, y);
		orig = Vec2f(x, y);
		weight = 1;
		for (k = 0; k < p;) {
			tparticles[k].x0 = tparticles[k].xp = tparticles[k].x = x;
			tparticles[k].y0 = tparticles[k].yp = tparticles[k].y = y;
			tparticles[k].sp = tparticles[k].s = 1.0;
			tparticles[k].width = width;
			tparticles[k].height = height;
			tparticles[k].r = region;
			tparticles[k].n = nbins;
			tparticles[k].toEval = true;
			getNormalizedHistogram(tparticles[k].vhistos, nbins, patches, roih, roim);
			tparticles[k++].w = 0;
		}
		bool anyFG = isConsistent(mask);
		if (!anyFG) {
			cout << "No FG." << endl;
		}
	}

	float likelihood(Mat& img, Mat& mask, Rect rec,
			float vh[][NH * NS + NV], int nrowcols, int n, int * count=0) {
		Mat tmp;
		float vhistos[NVV][NH * NS + NV];
		rec = rec & world;
		int width = rec.width;
		int height = rec.height;
		if (height <= 3 || width <= 3)
			return 0;
		int hpatchArea = (width / 2) * (height / nrowcols);
		int vpatchArea = (height / 2) * (width / nrowcols);
		if (hpatchArea < minArea / 2 && vpatchArea < minArea / 2)
		{
			//cerr << "rec.height*rec.width/nrowcols*nrowcols="<<(rec.height*rec.width)/(nrowcols*nrowcols)<<",minArea="<< minArea << endl;
			return 0;
		}
//		if(rec.height*rec.width/(nrowcols*nrowcols)<minArea/3)
//		{
//			cerr << "rec.height*rec.width="<<(rec.height*rec.width)<<", nrowcols="<<nrowcols<<",minArea="<< minArea << endl;
//			return 0;
//		}


		if(rec.width<nrowcols || rec.height<nrowcols)
			return 0;
		/* extract region around (r,c) and compute and normalize its histogram */
		Mat imgRoi = Mat(img, rec);
		vector < Rect > localPatches;

		u.fixedPatches(rec.height, rec.width, nrowcols, localPatches);
		tmp = imgRoi.clone();
		Mat mroi = Mat(mask,rec);
		//int countContours;
		//Rect lr;
		//mroi = getLargestContour(mroi,lr, &countContours);

		bool anyFG = isConsistent(mask, rec);
		if (!anyFG)
		{
//			Mat maskc = mask.clone();
//			rectangle(maskc,rec,Scalar(200),1);
//			imshow("mask",maskc);
//			waitKey(1);
			return 0;
		}
		if(count!=0)
			*count = getNormalizedHistogram(vhistos, n, localPatches, tmp, mroi);
		else
			getNormalizedHistogram(vhistos, n, localPatches, tmp, mroi);

		vector<float> compareVec;
		for (uint i = 0; i < localPatches.size(); i++) {
			float result = 1 - compare_histograms_ks(&vhistos[i][0], &vh[i][0], n);
			if(result>0)
				compareVec.push_back(result);
			else
				cerr<< "result<=0" << endl;
		}
		float med=0;
		if(compareVec.size()>0)
			med = median(compareVec);

		return med;
	}


	float likelihood(Mat& img, Mat& mask, int r, int c, int w, int h,
			float* rhisto, int n) {
		Mat tmp;
		float result;
		float histo[NH * NS + NV];
		if (h <= 3 || w <= 3)
			return 0;

		/* extract region around (r,c) and compute and normalize its histogram */
		Rect rec(c - w / 2, r - h / 2, w, h);
		rec = rec & world;
		Mat imgRoi = Mat(img, rec);
		//		rectangle(mask,rec,Scalar(200),1);
		//		imshow("Rect", mask);
		Mat mroi(mask, rec);
		tmp = imgRoi.clone();
		bool anyFG = isConsistent(mask);
		if (!anyFG)
			return 0;

		getNormalizedHistogram(histo, n, tmp, mroi);
		//normalize_vhist( histo );
		/* compute likelihood as e^{\lambda D^2(h, h^*)} */
		//		d_sq = histo_dist_battacharyya(histo, rhisto, n);
		//		float result = exp(-LAMBDA * d_sq);
		result = 1 - compare_histograms_ks(histo, rhisto, n);
		return result;
	}

	Mat getLargestContour(Mat & mask, Rect & largestRect, vector<Point> & lvp, int * count=0)
	{
		vector<vector<Point> > contours;
		Mat largestContourMask;
		//Mat contImg = mask.clone();
		Mat contImg = Mat::zeros(mask.rows+2, mask.cols+2,CV_8U);
		Rect rroi(1,1,mask.cols,mask.rows);
		Mat tmproi(contImg,rroi);
		mask.copyTo(tmproi);
		vector<Vec4i> hierarchy;
		int largestArea=0;
		int largestIndex=0;
		findContours( contImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		for( uint i = 0; i < contours.size(); i++ )
		{
			int area = contourArea(contours[i]);
			if(area>largestArea)
			{
				largestArea=area;
				largestIndex=i;
			}
		}

		if(count!=0)
			*count=contours.size();

		if(contours.size())
		{
			for(uint j=0; j<contours[largestIndex].size(); j++)
			{
				contours[largestIndex][j].x--;
				contours[largestIndex][j].y--;
			}
			largestRect=boundingRect(Mat(contours[largestIndex]));
			Mat tmpCont =Mat::zeros(mask.size(),CV_8U);
			drawContours(tmpCont,contours,largestIndex,Scalar(255),CV_FILLED );
			largestContourMask=Mat(tmpCont,largestRect);
			lvp=contours[largestIndex];
		}

		return largestContourMask.clone();
	}

	float histo_dist_battacharyya(float* hist1, float* hist2, int n) {
		//  float* hist1, * hist2;
		float sum = 0;
		int i;
		//  n = h1->n;
		//  hist1 = h1->histo;
		//  hist2 = h2->histo;
		/*
		 According the the Battacharyya similarity coefficient,
		 D = \sqrt{ 1 - \sum_1^n{ \sqrt{ h_1(i) * h_2(i) } } }
		 */
		for (i = 0; i < n; i++)
			sum += sqrt(hist1[i] * hist2[i]);
		return 1.0 - sum;
	}
	particle callTransition(particle p, int w, int h, gsl_rng* rng) {
		float x, y, s;
		particle pn;
		/* sample new state using second-order autoregressive dynamics */
		x = A1 * (p.x - p.x0) + A2 * (p.xp - p.x0)
				+ B0 * gsl_ran_gaussian(rng, TRANS_X_STD) + p.x0;
		pn.x = MAX(0.0, MIN((float) w - 1.0, x));
		y = A1 * (p.y - p.y0) + A2 * (p.yp - p.y0)
				+ B0 * gsl_ran_gaussian(rng, TRANS_Y_STD) + p.y0;
		pn.y = MAX(0.0, MIN((float) h - 1.0, y));
		s = A1 * (p.s - 1.0) + A2 * (p.sp - 1.0)
				+ B0 * gsl_ran_gaussian(rng, TRANS_S_STD) + 1.0;
		pn.s = MAX(0.1, s);
		pn.xp = p.x;
		pn.yp = p.y;
		pn.sp = p.s;
		pn.x0 = p.x0;
		pn.y0 = p.y0;
		pn.width = p.width;
		pn.height = p.height;
		pn.n = p.n;
		if (p.n > 400 || p.n < 50)
			cerr << "error: " << p.n << endl;

		//pn.patches = p.patches;
		memcpy(&pn.vhistos[0][0], &p.vhistos[0][0], pn.n * 4 * NVV);
		pn.w = 0;
		Rect r(MAX(0, pn.x - pn.width / 2), MAX(0, pn.y - pn.height / 2),
				pn.width, pn.height);
		//		int area=r.width*r.height;
		//		if(area<minArea)
		//			cout << "Transit. Before " << area << endl;
		r = r & world;
		//		area=r.width*r.height;
		//		if(area<minArea)
		//			cout << "Transit. After " << area << endl;
		pn.r = r;
		pn.toEval=true;
		return pn;
	}
	/*
	 Normalizes particle weights so they sum to 1

	 @param particles an array of particles whose weights are to be normalized
	 @param n the number of particles in \a particles
	 */
	void normalize_weights() {
		float sum = 0;
		int i;
		for (i = 0; i < np; i++)
			if(tparticles[i].w>0)
				sum += tparticles[i].w;
		for (i = 0; i < np; i++)
			tparticles[i].w /= sum;
	}
	int isConsistent(Mat& frame) {
		Rect r0 = tparticles[0].r;
		Mat roi(frame, r0);
		int count = cv::countNonZero(roi);
		return count;
	}

	bool isConsistent(Mat& frame, Rect& r) {
		Mat roi(frame, r);
		int count = cv::countNonZero(roi);
		if (count != 0)
			return true;
		else
			return false;
	}
	/*
	void addRegions(Mat& frame, Mat& mask, vector<Rect>& regions) {
		int width;
		int height;
		float x, y;
		int np1 = np - 1;
		int np1i;
		for (uint i = 0; i < regions.size(); i++) {
			width = regions[i].width;
			height = regions[i].height;
			x = regions[i].x + (float) ((((width / 2))));
			y = regions[i].y + (float) ((((height / 2))));
			Mat mroi = Mat(mask, regions[i]);
			Mat mframe = Mat(frame, regions[i]);
			int anyFG = isConsistent(mask, tparticles[0].r);
			if (!anyFG) {
				continue;
			}
			np1i = np1 - i;
			//#ifdef USE_PATCHES
			//getNormalizedHistogram(tparticles[np1].vhistos,tparticles[np1i].n,patches,roih);
			//#else
			getNormalizedHistogram(&tparticles[np1].histog[0],
					tparticles[np1i].n, mframe, mroi);
			//#endif
			tparticles[np1i].x0 = tparticles[np1i].xp = tparticles[np1i].x = x;
			tparticles[np1i].y0 = tparticles[np1i].yp = tparticles[np1i].y = y;
			tparticles[np1i].sp = tparticles[np1i].s = 1.0;
			tparticles[np1i].width = width;
			tparticles[np1i].height = height;
			tparticles[np1i].r = regions[i];
		}

	}
	*/
	void updateParticlesHistogram(Mat& img, Mat & mask, Rect region=Rect()) {
		if(region.width==0 && region.height==0)
			region = tparticles[0].r;

		int oldrc = nrows_cols;
		int width = region.width;
		int height = region.height;
		int hpatchArea = (width / 2) * (height / nrows_cols);
		int vpatchArea = (height / 2) * (width / nrows_cols);
		if (hpatchArea > minArea / 2 && vpatchArea > minArea / 2)
			return;

		int area = width * height;
		nrows_cols = sqrt(area / minArea);
		if (nrows_cols == 0)
			nrows_cols = 1;

		if (nrows_cols == oldrc)
			return;

		if (width <= 2 || height <= 2)
			nrows_cols = 1;

		if (nrows_cols > 8)
			nrows_cols = 8;

		vector < Rect > newPatches;
		u.fixedPatches(height, width, nrows_cols, newPatches);
		Mat mframe = Mat(img, region);
		Mat mroi = Mat(mask, region);

		for (int j = 0; j < np; j++) {
			getNormalizedHistogram(tparticles[j].vhistos, tparticles[j].n,
					newPatches, mframe, mroi);
		}
	}

	void updateObjDiagonal()
	{
		objectDiagonal=sqrt(observedRect.height * observedRect.height + observedRect.width * observedRect.height);
	}

	bool changeLastParticle(Mat& frame, Mat& mask, Rect& rn) {
		Rect forRect;
		forRect.x = rn.x + (int) (((incx)));
		forRect.y = rn.y + (int) (((incy)));
		forRect.width = rn.width + (int) (((incw)));
		forRect.height = rn.height + (int) (((inch)));
		forRect = forRect & world;
		int width = forRect.width;
		int height = forRect.height;
		int area = width * height;
		float x = forRect.x + (float) ((((width / 2))));
		float y = forRect.y + (float) ((((height / 2))));
		int np1 = np - 1;
		Mat mroi = Mat(mask, forRect);
		Mat mframe = Mat(frame, forRect);
		int anyFG = isConsistent(mask, forRect);
		if (!anyFG) {
			//cout << "No FG." << endl;
			return false;
		}
		if (area < minArea) {
			return true;
		}
		vector < Rect > localPatches;
		int hnrc = width / (float)sqrtMinArea, vnrc = height / (float)sqrtMinArea;
		int oldnrc = nrows_cols;

		if(min(hnrc, vnrc)>nrows_cols)
			nrows_cols = min(hnrc, vnrc);

		if (nrows_cols < 1 || width <= 2 || height <= 2)
			nrows_cols = 1;

		if (nrows_cols > 8)
			nrows_cols = 8;

		if(nrows_cols > width)
			nrows_cols = width;

		if(nrows_cols > height)
			nrows_cols = height;

		u.fixedPatches(height, width, nrows_cols, localPatches);
		if (oldnrc != nrows_cols) {
			for (int j = 0; j < np; j++) {
				getNormalizedHistogram(tparticles[j].vhistos, tparticles[j].n,
						localPatches, mframe, mroi);
			}
		}

		getNormalizedHistogram(tparticles[np1].vhistos, tparticles[np1].n,
				localPatches, mframe, mroi);
		tparticles[np1].x0 = tparticles[np1].xp = tparticles[np1].x = x;
		tparticles[np1].y0 = tparticles[np1].yp = tparticles[np1].y = y;
		tparticles[np1].sp = tparticles[np1].s = 1.0;
		tparticles[np1].width = width;
		tparticles[np1].height = height;
		tparticles[np1].r = forRect;
		return true;
	}

	/*
	 Re-samples a set of particles according to their weights to produce a
	 new set of unweighted particles

	 @param particles an old set of weighted particles whose weights have been
	 normalized with normalize_weights()
	 @param n the number of particles in \a particles

	 @return Returns a new set of unweighted particles sampled from \a particles
	 */
	void resample() {
		particle* new_particles;
		int i, j, k = 0;
		int nthis;
		//lastObservedRect = Rect(currObservedRect);
		//current = Rect(tparticles[0].r);
		//currObservedRect = Rect(tparticles[0]);
		qsort(tparticles, np, sizeof(particle), &particle_cmpHT);
		new_particles = (particle*) (((malloc(np * sizeof(particle)))));
		for (i = 0; i < np; i++) {
			nthis = cvRound(tparticles[i].w * np);
			for (j = 0; j < nthis; j++) {
				new_particles[k++] = tparticles[i];
				if (k == np)
					goto exit;
			}
		}

		while (k < np)
			new_particles[k++] = tparticles[0];
		exit: free(tparticles);
		tparticles = new_particles;
		evalDiff(observedRect, lastObservedRect);
	}
	void evalDiff(Rect& curr, Rect& past) {
		if(matchCount<1 || !match)
			return;

		int lastp=cmtrackx.size()-1;
		Vec2f p1(cmtrackx[lastp],cmtracky[lastp]);
		float disp=calcDisplacement(orig,p1);
		if(disp>perwdiag)
		{
			acumDisplacement+= disp;
			orig=p1;
			displTries=0;
		}
		else
			displTries++;

		if(displTries>20)
			acumDisplacement*=0.9;
	}

	float calcDisplacement(Point& orig, Point & par0) {
		float disp = sqrt(
				pow((float) (((orig.x))) - par0.x, 2)
						+ pow((float) (((orig.y))) - par0.y, 2));
		return disp;
	}
	float calcDisplacement(Point& orig, Rect& par0) {
		float disp = sqrt(
				pow((float) (((orig.x))) - par0.x, 2)
						+ pow((float) (((orig.y))) - par0.y, 2));
		return disp;
	}

	float calcDisplacement(Vec2f& orig, Vec2f& nposition) {
		float disp = sqrt(
				pow(orig[0] - nposition[0], 2)
						+ pow(orig[1] - nposition[1], 2));
		return disp;
	}

	float compareIntExtHistograms(Mat& roiF, Mat& roiM) {
		int maxbins = NH * NS + NV;
		float hobj[maxbins];
		float hreg[maxbins];
		int nbins = roiF.cols * roiF.rows;
		nbins = max(8, nbins);
		nbins = min(nbins, maxbins);
		getNormalizedHistogram(hobj, nbins, roiF, roiM);
		getNormalizedHistogram(hreg, nbins, roiF, roiM == 0);
		//		printArray(hobj,nbins);
		//		printArray(hreg,nbins);
		float ks = 1 - compare_histograms_ks(hobj, hreg, nbins);
		return ks;
	}

	float compareIntExtHistogramsIH(Mat& roiF, Mat& roiM, vector < Mat > & IIV_I) {
		int maxbins = NH * NS + NV;
//		float hobj[maxbins];
//		float hreg[maxbins];
		int nbins = roiF.cols * roiF.rows;
		nbins = max(8, nbins);
		nbins = min(nbins, maxbins);
		vector < float > hobj(maxbins);
		vector < float > hreg(maxbins);

		u.compute_histogram( 0, 0, roiF.rows-1, roiF.cols-1, &IIV_I , hobj );
		u.compute_histogram( 0, 0, roiF.rows-1, roiF.cols-1, &IIV_I , hreg );

		//getNormalizedHistogram(hobj, nbins, roiF, roiM);
		//getNormalizedHistogram(hreg, nbins, roiF, roiM == 0);
		//		printArray(hobj,nbins);
		//		printArray(hreg,nbins);
		float ks = 1 - compare_histograms_ks((float *)&hobj[0], (float *)&hreg[0], nbins);
		return ks;
	}


	//	void updatePatches(Rect& r) {
	//		fixedPatches(r.height, r.width, (int) (sqrt(patches.size())), patches);
	//	}
	float compare_histograms_ks(float* h1, float* h2, int n) {
		float sum = 0;
		float cdf1 = 0;
		float cdf2 = 0;
		float z;
		float ctr = 0;
		vector<double>::iterator it1, it2;
		for (int i = 0; i < n; i++) {
			cdf1 = cdf1 + *h1++;
			cdf2 = cdf2 + *h2++;
			z = cdf1 - cdf2;
			sum += abs(z);
			ctr = ctr + 1;
		}
		return (sum / ctr);
	}

	void drawText(float number, float fontScale, Mat& src, uchar color=255) {
		int baseline = 0;
		int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
		int thickness = 1;
		baseline += thickness;
		Point textOrg(0, src.rows / 2);
		string text;
		stringstream ss; //create a stringstream
		ss << number; //add number to the stream
		text = ss.str(); //return a string with the contents of the stream
		putText(src, text, textOrg, fontFace, fontScale, Scalar(color, 0, 0),
				thickness, 8);
	}

	float compareVicinityHistograms(Mat& roiF, Mat& roiM, Mat& dr, Mat& er,
			Mat& rgbCont) {
		int maxbins = NH * NS + NV;
		//		Mat dr, er;
		//		Size sd = Size(3,3);
		//		dilate(roiM,dr,getStructuringElement(MORPH_RECT,sd));
		//		erode(roiM,er,getStructuringElement(MORPH_RECT,sd));
		Mat dil, ero;
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
		dilate(dr, dil, element);
		erode(er, ero, element);
		Mat external, internal;
		absdiff(dil, dr, external); //absdiff(dil,roiM,external);
		absdiff(ero, er, internal); //absdiff(ero,roiM,internal);
		float hobj[maxbins];
		float hreg[maxbins];
		int area = roiF.cols * roiF.rows;
		int nbins = area;
		if (nbins < maxbins)
			return 0;

		nbins = max(8, nbins);
		nbins = min(nbins, maxbins);
		getNormalizedHistogram(hobj, nbins, roiF, internal);
		getNormalizedHistogram(hreg, nbins, roiF, external);
		float ks = 1 - compare_histograms_ks(hobj, hreg, nbins);
		//		imshow("internal",internal);
		//		imshow("external",external);
		//		waitKey();
		return ks;
	}

	void cropMat(Mat & src, Rect & reference)
	{
		vector<Point> vp;
		int c;
//		imshow("src",src);
		src = getLargestContour(src,reference,vp,&c);
//		imshow("cropped",src);
		//waitKey(0);
	}

	float compareVicinityHistogramsPatches(Mat& roiF, Mat& roiM, Mat& dr,
			Mat& er, int nRowCol, vector<Point> contour, float * f3a, float * f4a, Mat rgbCont=Mat(), float upth=0.99, float difth=0.2, float disrate=0.85) {
		int maxbins = NH * NS + NV;
		//float upth=0.99,

		int blobArea = countNonZero(roiM);
		if(blobArea<600)
		{
			upth=upth - difth;
		}
		float lowth=upth - difth;
		vector<float> mresultu;
		vector<float> mresultl;
		//		Mat dr, er;
		//		Size sd = Size(3,3);
		//		dilate(roiM,dr,getStructuringElement(MORPH_RECT,sd));
		//		erode(roiM,er,getStructuringElement(MORPH_RECT,sd));
		Mat external, internal;
		Mat dil, ero;
		vector < Rect > pcs;
		float hobj[maxbins];
		float hreg[maxbins];
		int area = roiF.cols * roiF.rows;
		int nbins = area;
		if (nbins < maxbins)
			return 0;
		nbins = max(8, nbins);
		nbins = min(nbins, maxbins);
		vector<Mat> vecext(3), vecint(3), vecroiF(3);
		vector < vector < Rect > > vecpcs(3);
		int chosen=-1;
		uint bigvecsize=0;
		Mat rroiF;
		for(int it=0; it<2; it++)
		{
			Mat element = getStructuringElement(MORPH_RECT,
					Size((nRowCol+it) * 4, (nRowCol+it) * 4));
			dilate(roiM, dil, element);
			erode(roiM, ero, element);
			Rect refR;
			cropMat(dil,refR);
			Mat roiEro=Mat(ero,refR);
			Mat roiEr=Mat(er,refR);
			Mat roiDr=Mat(dr,refR);
			Mat rroiF=Mat(roiF,refR);
//			Mat roiEro=ero;
//			Mat roiEr=er;
//			Mat roiDr=dr;
//			rroiF=roiF;
			absdiff(dil, roiDr, external); //absdiff(dil,roiM,external);
			absdiff(roiEro, roiEr, internal); //absdiff(ero,roiM,internal);
			vector<float> psizes={(float)nRowCol,(float)(nRowCol+1)};
			vector<float> weights;
			vector<Rect> currpcs;//=findBestPatches(psizes,internal,external,contour,disrate,weights);
			if(currpcs.size()>0)
			{
				vecpcs[it]=currpcs;
				vecext[it]=external;
				vecint[it]=internal;
				vecroiF[it]=rroiF;
				if(currpcs.size()>bigvecsize)
				{
					bigvecsize=currpcs.size();
					chosen=it;
				}
			}

#ifdef DEBUG
		Mat rgbcontclone = rroiF.clone();
#endif
			for (uint i = 0; i < pcs.size(); i++) {
				Mat pRoiF(rroiF, pcs[i]);
				Mat pRoiMI(internal, pcs[i]);
				Mat pRoiME(external, pcs[i]);
				Mat pRoiM(roiM, pcs[i]);
				getNormalizedHistogram(hobj, nbins, pRoiF, pRoiMI);
				getNormalizedHistogram(hreg, nbins, pRoiF, pRoiME);
				float ks = 1 - compare_histograms_ks(hobj, hreg, nbins);
				if(ks>=upth)
					mresultu.push_back(ks);
				else if(ks<lowth)
					mresultl.push_back(ks);
				//mresult.push_back(ks);
//	#ifdef DEBUG
//				cout << "F4_ks = " << ks << endl;
//				rectangle(external,pcs[i],Scalar(150),1);
//				rectangle(internal,pcs[i],Scalar(150),1);
//				rectangle(rgbcontclone,pcs[i],Scalar(150,0,255),1);
//	#endif
			}
		}
		if(chosen>=0)
		{
			pcs=vecpcs[chosen];
			internal=vecint[chosen];
			external=vecext[chosen];
			rroiF=vecroiF[chosen];
		}

#ifdef DEBUG
//		drawText(ID, 0.45, rgbCont);
		//imshow("F4_rgb",rgbcontclone);
//		imshow("external",external);
//		imshow("internal",internal);
//		imshow("roiM",roiM);
//		waitKey(0);
#endif
		float featureValue = -1;
		int msize=(mresultu.size() + mresultl.size());
		if(msize>0)
		{
			featureValue = mresultu.size()/(float)msize;
			*f4a = *f4a*alphaF + featureValue*(1-alphaF);
		}
		else
		{
//			cout << "missing value.f4Average=" << *f4a << endl;
			featureValue = *f4a;
		}
#ifdef DEBUG
		cout << "f4=" << featureValue << ", msize=" << msize << endl;
#endif
		return featureValue;
	}

	vector < Rect > findBestPatches(vector<Size> psize, Mat& r1, Mat& r2, vector<Point> contour, float disrate, vector<float> weights)
	{
		float rate;
		RNG rng;

		vector<int> countValid(psize.size());
		vector< vector < Rect > > vecpatches(psize.size());
		vector < Rect > validpcs;
		for(uint ps=0; ps<psize.size(); ps++)
		{
			vector < Rect > pcs;
//			float nps=psize[ps];
//			if(nps<=0)
//				continue;

			u.fixedPatches(r1.rows, r1.cols, psize[ps], pcs);
			//u.randomPatches( r1.rows, r1.cols,psize[ps], contour,pcs);
			for (uint i = 0; i < pcs.size(); i++) {
				Mat r1RoiM(r1, pcs[i]);
				Mat r2RoiM(r2, pcs[i]);
				int r1NonZero = countNonZero(r1RoiM);
				int r2NonZero = countNonZero(r2RoiM);
				rate = 0;
				int minn=MIN(r1NonZero,r2NonZero);
				int maxn=MAX(r1NonZero,r2NonZero);
				if(maxn>0)
					rate= minn / (float)maxn;
				if (rate > disrate || rate < (1-disrate) || minn<30)
					continue;
				validpcs.push_back(pcs[i]);
				weights.push_back(rate);
			}
		}
		return validpcs;
	}

	int getGoalLength(int len, int smallLen, int goalArea)
	{
		int area=len*smallLen;
		int newLen=area/goalArea;
		return newLen;
	}

	vector<Size> getLargerGrid(Size g, Rect r)
	{
		Size g1(g.width-1,g.height);
		Size g2(g.width,g.height-1);
		Size g3(g.width-2,g.height);
		Size g4(g.width,g.height-2);

		vector<Size> vs;
		vs.push_back(g);
		for(int i=g.width; i>=g.width-6; i--)
		{
			for(int j=g.height; j>=g.height-6; j--)
			{
				if(i>1 && j>1)
					vs.push_back(Size(i,j));
			}
		}

		vector<float> diffs;
		for(uint i=0; i<vs.size(); i++)
		{
			int w1=r.width/vs[i].width;
			int h1=r.height/vs[i].height;
			//float d=sqrt(w1*w1+h1*h1);
			float diff=abs((float)w1-(float)h1);
			diffs.push_back(diff);
		}
		vector<int> perm = u.sort_permutation(diffs);
//		u.printVector(diffs);
//		u.printVector(perm);

		vector<Size> orderedSizes=u.apply_permutation(vs,perm);
		//vector<Size> resultSizes;
		if(orderedSizes.size()>10)
			orderedSizes.erase(orderedSizes.begin()+10,orderedSizes.end());
		return orderedSizes;
	}

	float compareIntExtHistogramsPatches(Mat& roiF, Mat& roiM, Mat& dr, Mat& er,
			Size grid, vector<Point> contour, float * f3a, float * f4a, Mat & rgbCont, float upth=0.96, float difth=0.2, float disrate=0.85) {
		int maxbins = NH * NS + NV;
		vector<float> mresultu;
		vector<float> mresultl;
#ifdef DEBUG
		Mat rgbcontclone = rgbCont.clone();
#endif
		//float upth=0.96,
//		int blobArea = countNonZero(roiM);
//		if(blobArea<600)
//		{
//			upth=upth - difth;
//		}

		float lowth=upth - difth;
		float hobj[maxbins];
		float hreg[maxbins];
		int area = roiF.cols * roiF.rows;
		int nbins = area;
		if (nbins < maxbins)
			return 0;

		nbins = max(8, nbins);
		nbins = min(nbins, maxbins);
		vector < Rect > pcs, pcsrec;

		vector<Size> psizes; //={nRowCol-2,nRowCol-1,nRowCol};
		psizes = getLargerGrid(grid,Rect(0,0,roiF.cols,roiF.rows));
		Mat invdr=dr==0;
//		Mat border=Mat::zeros(roiM.size(),CV_8U);
//		vector<vector<Point> > vcontour;
//		vcontour.push_back(contour);
//		drawContours(border,vcontour,0,Scalar(150));
//		std::vector<cv::Point> locations;   // output, locations of non-zero pixels
//		findNonZero(border, locations);
		//vector<Point> vecp = u.sampleBorder(border,er);
//		Mat points=Mat::zeros(roiM.size(),CV_8U);
//		u.drawPoints(locations,points,Scalar(150));
//		imshow("points",points);
//		waitKey(0);
		vector<float> weights;
		pcs=findBestPatches(psizes,roiM,invdr,contour,disrate,weights);

		for (uint i = 0; i < pcs.size(); i++) {
			Mat pRoiF(roiF, pcs[i]);
			Mat objRoiM(er, pcs[i]);
			Mat regRoiM(dr, pcs[i]);

			getNormalizedHistogram(hobj, nbins, pRoiF, objRoiM);
			getNormalizedHistogram(hreg, nbins, pRoiF, regRoiM == 0);
			float ks = 1 - compare_histograms_ks(hobj, hreg, nbins);

			if(ks>=upth)
				mresultu.push_back(ks);
			else if(ks<lowth)
				mresultl.push_back(ks);
#ifdef DEBUG
	//			cout << "F3_ks = " << ks << endl;
				//Mat mcontRoi(rgbcontclone,pcs[i]);
				rectangle(rgbcontclone,pcs[i],Scalar(0,255,0),1);
				rectangle(dr,pcs[i],Scalar(150),1);
				rectangle(er,pcs[i],Scalar(150),1);
#endif
		}
#ifdef DEBUG
		imshow("rgbF3",rgbcontclone);
		imshow("drF3",dr);
		imshow("erF3",er);
//		waitKey(0);
#endif
			float featureValue = -1;
			int mmsize=(mresultu.size() + mresultl.size());
			if(mmsize>0)
			{
				featureValue = mresultu.size()/(float)mmsize;
				*f3a = *f3a*alphaF + featureValue*(1-alphaF);
			}
			else
			{
				//cout << "missing value. f3Average=" << *f3a << endl;
				featureValue = *f3a;
			}
#ifdef DEBUG
		cout << "f3=" << featureValue << "," << "mmsize=" << mmsize << endl;
//		if(featureValue==0)
//		{
//			cout << "showing F3." << endl;
//			waitKey(0);
//		}
#endif
			return featureValue;
	}

	float weightedMedian(vector<float> x, vector<float> w) {
		//size_t N = x.size();
		//	  double x[N] = {0, 1, 2, 3, 4};
		//	  double w[N] = {.1, .2, .3, .4, .5};
		sort(w.begin(), w.end());
		double S = std::accumulate(w.begin(), w.end(), 0.0); // the total weight
		int k = 0;
		double sum = S - w[0]; // sum is the total weight of all `x[i] > x[k]`
		while (sum > S / 2) {
			++k;
			sum -= w[k];
		}
		return x[k];
	}

	double median(vector<float> vec) {
		typedef vector<float>::size_type vec_sz;
		vec_sz size = vec.size();
		if (size == 0)
			//throw domain_error("median of an empty vector");
			cerr << "median of an empty vector" << endl;

		sort(vec.begin(), vec.end());
		vec_sz mid = size / 2;
		return size % 2 == 0 ? (vec[mid] + vec[mid - 1]) / 2 : vec[mid];
	}


	float compareHistograms(Mat& img1, Mat& mask1, Mat& img2, Mat& mask2) {
		int maxbins = NH * NS + NV;
		float hobj1[maxbins];
		float hobj2[maxbins];
		int nbins1 = img1.cols * img1.rows;
		int nbins2 = img2.cols * img2.rows;
		nbins1 = max(8, nbins1);
		nbins2 = max(8, nbins2);
		nbins1 = min(nbins1, maxbins);
		nbins2 = min(nbins2, maxbins);
		int nbins = max(nbins1, nbins2);
		getNormalizedHistogram(hobj1, nbins1, img1, mask1);
		getNormalizedHistogram(hobj2, nbins2, img2, mask2);
		float d_sq = histo_dist_battacharyya(hobj1, hobj2, nbins);
		float result = exp(-LAMBDA * d_sq);
		return result;
	}

	bool isParticle0AtBorder() {
		bool atborder = tparticles[0].r.x < 2 || tparticles[0].r.y < 2
				|| (tparticles[0].r.x + tparticles[0].r.width) > world.width - 2
				|| (tparticles[0].r.y + tparticles[0].r.height)
						> world.height - 2;
		return atborder;
	}

	bool atBorder()
	{
		bool atborder = rectAtBorder(observedRect);
		return atborder;
	}

	bool rectAtBorder(Rect& r) {
		bool atborder = (r.x <= minDiagDistance || r.y <= minDiagDistance
				|| (r.x + r.width) >= world.width - minDiagDistance
				|| (r.y + r.height) >= world.height - minDiagDistance);
		return atborder;
	}

	//	float getAverageChanges()
	//	{
	//		return (abs(incx) + abs(incy) + abs(incw) + abs(inch))/4;
	//	}
	void setWorld(const Rect& world) {
		this->world = world;
		diagonal = sqrt(
				world.height * world.height + world.width * world.height);
		perwdiag = 0.005 * diagonal;
		oneTenthDiag = 0.1 * diagonal;
		minDiagDistance = 2;//MIN(0.02 * diagonal,4);
		if (perwdiag < 2)
			perwdiag = 2;
	}
	bool getDisplaced() {
		psr16  = MIN(1,acumDisplacement / (16 * objectDiagonal));
		displaced = acumAreaChanges>24 || acumDisplacement>30 || psr16>0.03;
		return (displaced);
	}

	int nonZero(vector<Point>& v, Mat& src) {
		int count = 0;
		for (uint i = 0; i < v.size(); i++) {
			if (src.at < uchar > (v[i]) != 0)
				count++;
		}
		return count;
	}

	int isObjectValidWithHull(Mat& src, Mat& mask, Mat& hull, bool lowStd, Mat tdiff =
			Mat()) {
		Mat dil, ero;
		Mat sobedges = src.clone();
		Mat drawCont = src.clone();
		u.applySobel(sobedges);
		Size sd = Size(13, 13);
		dilate(hull, dil, getStructuringElement(MORPH_ELLIPSE, sd));
		Size se = Size(7, 7);
		erode(mask, ero, getStructuringElement(MORPH_ELLIPSE, se));
		Mat coroa;
		coroa = dil & (ero == 0);
		Mat sobCoroa, canCoroa;
		cvtColor(sobedges, sobedges, CV_BGR2GRAY);
		Mat sumEdges;
		Mat sum8b;
		Mat sumth;
		sum8b = sobedges & coroa;
		Scalar MED = mean(sum8b);
		Scalar MAD = abs(mean(MED - sum8b).val[0]);
		int sumthrsh;
//		sumthrsh = (int) (MED.val[0] + 1.2 * 1.4826 * MAD.val[0]);
		if(lowStd)
			sumthrsh = (int) (MED.val[0] + 1 * 1.4826 * MAD.val[0]);
		else
			sumthrsh = (int) (MED.val[0] + 2 * 1.4826 * MAD.val[0]);

		threshold(sum8b, sumth, sumthrsh, 255, THRESH_BINARY);

		int result = 1; //false;
		//imwrite("sumth.png", sumth);
		result = findObjects(sumth, drawCont);
		if (result > 0)
		{
			return result;
		}

		Mat diffEdges = tdiff;
		sumth = sumth | (diffEdges & coroa);
		result = findObjects(sumth, drawCont);
		if(result>0)
		{
//					imshow("sobedges.png", sobedges);
//					imshow("coroa.png", coroa);
//					imshow("drawCont",drawCont);
//					imshow("sumth",sumth);
			//waitKey(0);
		}
		return result;
	}
	int isObjectValid(Mat& src, Mat& mask, Mat tdiff = Mat()) {
		Mat dil, ero;
		//Mat canedges = src.clone();
		Mat sobedges = src.clone();
		Mat drawCont = src.clone();
		u.applySobel(sobedges);
		Size sd = Size(13, 13);
		dilate(mask, dil, getStructuringElement(MORPH_ELLIPSE, sd)); //
		Size se = Size(7, 7);
		erode(mask, ero, getStructuringElement(MORPH_ELLIPSE, se)); //
		Mat coroa;
		coroa = dil & (ero == 0);
		Mat sobCoroa, canCoroa;
		cvtColor(sobedges, sobedges, CV_BGR2GRAY);
		Mat sumEdges;
		Mat sum8b;
		Mat sumth;
		sum8b = sobedges & coroa;
		Scalar MED = mean(sum8b);
		Scalar MAD = abs(mean(MED - sum8b).val[0]);
		float sumthrsh = (int) (((MED.val[0] + 0.8 * 1.4826 * MAD.val[0])));
		threshold(sum8b, sumth, sumthrsh, 255, THRESH_BINARY);
		//		imshow("mask.png", mask);
		//		imshow("sobedges.png", sobedges);
		//		imwrite("coroa.png", coroa);
		//		imwrite("dil.png", dil);
		//		imwrite("ero.png", ero);
		//		imwrite("sumth.png", sumth);
		//		imwrite("sum8b.png", sum8b);
		int result = 1; //false;
		result = findObjects(sumth, drawCont);
		if (result > 0)
			return result; //true;

		Mat diffEdges = tdiff;
		sumth = sumth | diffEdges;
		result = findObjects(sumth, drawCont);
		return result;
	}
	int findObjects(Mat& sumth, Mat src) {
		Size s1 = sumth.cols > 55 && sumth.rows > 55 ? Size(7, 7) : Size(5, 5);
		morphologyEx(sumth, openEdges, MORPH_CLOSE,
				getStructuringElement(MORPH_ELLIPSE, s1));
		//int ch[] = { };
		int ch[] = { 0,0,1,1,2,2};
		vector < Mat > planes;
		vector < Mat > vres;
		edgesRGB = Mat(openEdges.size(), CV_8UC3);
		planes.push_back(edgesRGB);
		vres.push_back(openEdges);
		vres.push_back(openEdges);
		vres.push_back(openEdges);
		mixChannels(vres, planes, ch, 3);
		edgesRGB = planes[0];
		vector < vector<Point> > contours;
		vector < vector<Point> > opencontours;
		vector < Vec4i > hierarchy;
		Mat expEdges = Mat::zeros(openEdges.rows + 2, openEdges.cols + 2,
				CV_8U);
		Rect r(1, 1, openEdges.cols, openEdges.rows);
		Mat roi(expEdges, r);
		openEdges.copyTo(roi);
		Mat expRGB = Mat(expEdges.size(), CV_8UC3);
		findContours(expEdges, contours, hierarchy, CV_RETR_TREE,
				CV_CHAIN_APPROX_SIMPLE);
		if (contours.empty() || hierarchy.empty())
			return -1; //false;

		int biggestContour = -1;
		int biggestArea = 0;
		int currArea = 0;
		int cvecsize = 0;
		for (uint i = 0; i < contours.size(); i++) {
			if (hierarchy[i][2] != -1) {
				Rect r0 = boundingRect(Mat(contours[hierarchy[i][2]]));
				if (r0.width > 0.05 * sumth.cols
						&& r0.height > 0.05 * sumth.rows)
					return 2; //true;
				else {
					Rect r1 = boundingRect(Mat(contours[i]));
					currArea = r1.width * r1.height;
					if (currArea > biggestArea) {
						biggestArea = currArea;
						biggestContour = cvecsize;
					}
					opencontours.push_back(contours[i]);
					cvecsize++;
				}

			} else {
				Rect r1 = boundingRect(Mat(contours[i]));
				currArea = r1.width * r1.height;
				if (currArea > biggestArea) {
					biggestArea = currArea;
					biggestContour = cvecsize;
				}
				opencontours.push_back(contours[i]);
				cvecsize++;
			}

		}

		if (!opencontours.empty()) {
			return hasSmallAngleDefects(opencontours[biggestContour], edgesRGB,
					expRGB);
		} else
			return -2; //false;
	}
	int hasSmallAngleDefects(vector<Point>& v, Mat& edges, Mat& srcRGB) {
		int angDefects = 0;
		vector < Vec4i > defects;
		vector<int> hull;
		vector < vector<Point> > hullp(1);
		std::vector < std::vector<cv::Point> > contours_poly(1);
		approxPolyDP(cv::Mat(v), contours_poly[0], 2, true);
		convexHull(contours_poly[0], hull);
		if (hull.size() > 3) {
			convexityDefects(contours_poly[0], hull, defects);
			angDefects = findSmallAngleDefect(defects, contours_poly[0], edges);
			if (angDefects > 0)
				return angDefects; //3;//true;
			//			else
			//				cout << "returning false: hull.size()" << hull.size()<< endl;
		}
		return -3; //false;
	}

	int findSmallAngleDefect(vector<Vec4i>& def, vector<Point>& ocontour,
			Mat& draw) {
		int count = 0;
		int diaglen = diagonal / 4;
		int smallestAngle = 360;
		for (uint i = 0; i < def.size(); i++) {
			if (def[i][3] / 256 > diaglen)
				continue;

			int angle = angleBetween(ocontour[def[i][2]], ocontour[def[i][1]],
					ocontour[def[i][0]]);
			if (angle < 65) {
				count++;
				if (angle < smallestAngle)
					smallestAngle = angle;
			}
		}

		//imshow("EdgesRGB",draw);
		//cout << count << endl;
		if (smallestAngle != 360)
			return smallestAngle * 10;
		else
			return -1;
	}
	int angleBetween(Point tip, Point next, Point prev) {
		float c1 = atan2((float) (((next.x))) - tip.x, next.y - tip.y);
		float c2 = atan2((float) (((prev.x))) - tip.x, prev.y - tip.y);
		float c = abs(c1 - c2);
		int degrees = abs(round(c * 180 / PI));
		if (degrees > 180)
			degrees = 360 - degrees;

		return degrees;
	}
	void setDisplaced(bool displ) {
		displaced = displ;
	}

	void displayParticle(Mat& img, particle p, Scalar color) {
		int x0, y0, x1, y1;
		x0 = cvRound(p.x - 0.5 * p.s * p.width);
		y0 = cvRound(p.y - 0.5 * p.s * p.height);
		x1 = x0 + cvRound(p.s * p.width);
		y1 = y0 + cvRound(p.s * p.height);
		rectangle(img, Point(x0, y0), Point(x1, y1), color, 1, 8, 0);
	}

	void displayParticles(Mat& img, CvScalar nColor, CvScalar hColor,
			int param) {
		CvScalar color;
		if (param == SHOW_ALL)
			for (int j = np - 1; j >= 0; j--) {
				if (j == 0)
					color = hColor;
				else
					color = nColor;
				displayParticle(img, tparticles[j], color);
			}

		if (param == SHOW_SELECTED) {
			color = hColor;
			displayParticle(img, tparticles[0], color);
		}
	}

//	float getDisplacement() const {
//		return displacement;
//	}

//	const Mat& getOpenEdges() const {
//		return openEdges;
//	}
//
//	const Mat& getEdgesRgb() const {
//		return edgesRGB;
//	}

	void setObservedRect(const Rect& observRect) {
		lastObservedRect = observedRect;
//		orig = Vec2f(observedRect.x + observedRect.width / 2,
//				observedRect.y + observedRect.height / 2);
		this->observedRect = observRect;
		updateObjDiagonal();
	}

	void setLastObservedRect(const Rect& lastObservedRect) {
		this->lastObservedRect = lastObservedRect;
	}

	float getAcumDisplacement() const {
		return acumDisplacement;
	}

//	int getMrsize() const {
//		return mrsize;
//	}

	const vector<Point>& getContourHull() const {
		return contourHull;
	}

	void setContourHull(const vector<Point>& contour) {
		this->contourHull = contour;
	}

	bool isOld(int currFrame, int threshold) {
		return (currFrame - startFrame) > threshold ? true : false;
	}

	int getAge(int currFrame) {
		return currFrame - startFrame + 1;
	}

	int getMinArea() const {
		return minArea;
	}

	void setMinArea(int minArea, int patcharea) {
		this->minArea = minArea;
		this->sqrtMinArea = sqrt(minArea);
		this->patchArea = patcharea;
	}

	void setStartFrame(int bournFrame) {
		this->startFrame = bournFrame;
	}

	int getNrowsCols() const {
		return nrows_cols;
	}

	particle* tparticles;
	int objectID;
	int weight;
	bool match;
	int matchCount;
	int analyzedCount;
	int startFrame;
	//int lastMatchFrame;
	Rect lastObservedRect;
	Rect observedRect;
	int reason;
	float incx;
	float incy;
	float incw;
	float inch;
	int perwdiag;
	int oneTenthDiag;
	//int age;
	Vec2f orig;
	Vec2f pastOrig;
	Point2f cOfMass;
	float acumDisplacement;
	int displTries;
	bool displaced;
	float psr16;
	int countDisplaced;
	int diagonal;
	int objectDiagonal;
	int minDiagDistance;
	float stddev;
	Mat openEdges;
	Mat edgesRGB;
	Mat roiObj;
	vector<Point> contourHull;
	vector<Point> contourReal;
	//int objType;
	//vector<bool> matches;
	vector<int> lifeTrack;
	vector<vector<Point> > contourHistory;
	vector<double> cmtrackx;
	vector<double> cmtracky;
	Vec3f currline, fittedLine;
	//bool isLineFitted;
	//int countVotes;
	int notGhost;
	int area;
	int length;
	int objMinArea;
	int objMaxArea;
	int objOldArea;
	int countAreaChanges;
	float acumAreaChanges;
	int areaTries;
	map<int, bool> cleared;
	map<int, int> mapCount;
	vector< vector<float> > ftvectotal;
	vector< vector<float> > f5features;
	vector<float> vstdevs;
	vector<int> defectsVotes;
	float alphaF;

	vector<Point> getContourReal() const {
		return contourReal;
	}

	void setContourReal(vector<Point> contourReal) {
		this->contourReal = contourReal;
	}

	int getArea() const {
		return area;
	}

	void setArea(int area) {
		float rate;
		if(objOldArea>0)
			rate = abs(objOldArea-area)/(float)objOldArea;
		else
			rate=0;
		if(objOldArea==0 || rate>0.05 )
		{
			countAreaChanges++;
			if(acumAreaChanges<1000)
				acumAreaChanges+=pow(1+rate,countAreaChanges);
			objOldArea=area;
			areaTries=0;
		}
		else
			areaTries++;
		if(areaTries>20)
			acumAreaChanges*=0.9;
		this->area = area;
		objMinArea=MIN(objMinArea,area);
		objMaxArea=MIN(objMaxArea,area);
	}

	int getLength() const {
		return length;
	}

	void setLength(int length) {
		this->length = length;
	}

private:
	//histogram * ref_histo;
	Rect world;
	//vector<Rect> patches;
	//std::map<int, vector<Rect> > patchMap;
	int nrows_cols;
	int mrsize;
	int minArea;
	int patchArea;
	int sqrtMinArea;
	int np;
	UtilCpp u;
};
class  trajectory
{
public:
	trajectory(){

	}
	~trajectory(){
		free(object.tparticles);
	}


	HessParticle object;
	//Mat objMask;

};

};
#endif
