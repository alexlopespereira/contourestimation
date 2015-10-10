/*
 * GhostDetector.h
 *
 *  Created on: Feb 9, 2014
 *      Author: alex
 */

#ifndef GHOSTDETECTOR_H_
#define GHOSTDETECTOR_H_
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "UtilCpp.h"
//#include "db.h"
#include "HessParticle.h"
#include "cmath"
#include <stdio.h>
#include <gsl/gsl_qrng.h>

using namespace Util;
using namespace std;
using namespace cv;

namespace HessTracking
{
class GhostDetector {
public:
	GhostDetector(){
		totalNonBlackFrames=0;
		totalGhosts=0;
		totalNoGhosts=0;
		totalBlobsAnalyzed=0;
		createdHeader=false;
		initResizeFactor=0.4;
		incResizeFactor=0.1;
		initCurrpatcharea=200;
		incCurrpatcharea=150;
		initWeight=0.6;
		incWeight=0.8;
		initUpth=0.93;
		incUpth=0.03;
		initDifth=0;
		incDifth=0.03;
		initDisrate=0.85;
		incDisrate=0.05;
		difFrame=-1;
	    f3Aver=0.3;
	    f4Aver=0.25;
	}
	~GhostDetector(){
	}

	int countContours(Mat & mask)
	{
		vector<vector<Point> > contours;
		Mat contImg = Mat::zeros(mask.rows+2, mask.cols+2,CV_8U);
		Rect rroi(1,1,mask.cols,mask.rows);
		Mat tmproi(contImg,rroi);
		mask.copyTo(tmproi);
		vector<Vec4i> hierarchy;
		findContours( contImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		int count = contours.size();
		return count;
	}

	vector<vector<Point> >  getContours(Mat & mask, vector<Rect> & regions, vector<Point2f> & vcm, vector<vector<Point> > & meaningfulContours, vector<int> & vAreas, vector<int> & vLen)
	{
		vAreas.clear();
		regions.clear();
		vcm.clear();
		vLen.clear();
		vector<vector<Point> > contours;
		vector<vector<Point> > hullSet;
		Mat contImg = Mat::zeros(mask.rows+2, mask.cols+2,CV_8U);
		Rect rroi(1,1,mask.cols,mask.rows);
		Mat tmproi(contImg,rroi);
		mask.copyTo(tmproi);
		//Mat contImg2 = mask.clone();
		//		Mat contImg3C = u.convertTo3Channels(contImg);
		vector<Vec4i> hierarchy;
		findContours( contImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
		//Mat cm = Mat::zeros(mask.size(),CV_8U);

		int areaTh=minArea;//*0.9;

		for( uint i = 0; i < contours.size(); i++ )
		{
			int area = contourArea(contours[i]);

			if (area > areaTh)
			{
				for(uint j=0; j<contours[i].size(); j++)
				{
					contours[i][j].x--;
					contours[i][j].y--;
				}
				vector<Point> hull;
				convexHull( Mat(contours[i]), hull, false );
//				if(hull.size()<5)
//					continue;
				Rect r0 = boundingRect( Mat(contours[i]) );
				regions.push_back(r0);
#ifdef MOMENTS
				Moments mu = moments( contours[i], false );
				Point2f mc = Point2f( mu.m10/mu.m00, mu.m01/mu.m00);
				vcm.push_back(mc);
#endif
				int len = arcLength(contours[i], false);
				hullSet.push_back(hull);
				meaningfulContours.push_back(contours[i]);
				vAreas.push_back(area);
				vLen.push_back(len);
			}
//			else
//				cout << "discarding blob with area " << area << endl;
		}
		return hullSet;
	}

	vector<vector<Point> >  getLines(Mat & mask, int * slen=NULL, float dist=0.2, bool both_borders=false)
	{
		int minLength;
		vector<Rect> regions;
		vector<vector<Point> > meaningfulContours;
		regions.clear();
		vector<vector<Point> > contours;
		Mat contImg = Mat::zeros(mask.rows+2, mask.cols+2,CV_8U);
		Rect rroi(1,1,mask.cols,mask.rows);
		Mat tmproi(contImg,rroi);
		mask.copyTo(tmproi);
		vector<Vec4i> hierarchy;
		findContours( contImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

		minLength=MAX(5,(mask.cols+mask.rows)/8);//1/4 da média
		if(slen!=0)
			minLength=5;
		for( uint i = 0; i < contours.size(); i++ )
		{
			int len = arcLength(contours[i], false);
			if (len <= minLength)
				continue;

			RotatedRect rr = fitEllipse(contours[i]);
			float rate = rr.size.width/rr.size.height;
			Rect r0 = boundingRect( Mat(contours[i]) );
			Rect ref(0,0,mask.cols,mask.rows);
			bool atborder = atBorder(r0,ref,dist,both_borders);
			if (rate<0.2 && (atborder || slen!=0 ))
			{
				if(slen!=0)
					*slen=*slen+len;
				//cout << "rate= " << rate << " , " << "len= " << len << endl;
				for(uint j=0; j<contours[i].size(); j++)
				{
					contours[i][j].x--;
					contours[i][j].y--;
				}

				regions.push_back(r0);
				meaningfulContours.push_back(contours[i]);
			}
//			else
//				cout << "discarding blob with area " << area << endl;
		}
		return meaningfulContours;
	}


	bool compareEllipses(RotatedRect & e1, RotatedRect & e2, float displ, float th)
	{
		float difAngle=abs(e1.angle-e2.angle);
		difAngle=min(difAngle,180-difAngle);
		Size2f difS=e1.size-e2.size;
		float thDecd=th*0.85;
		//float angleRatio=difAngle/pEllipse.angle;
		Size2f sizeRatio;
		sizeRatio.width = abs(difS.width/e2.size.width);
		sizeRatio.height = abs(difS.height/e2.size.height);
		//float displ = sqrt(pow(trajectories[tr]->object.cOfMass.x - vcmass[i].x,2)+pow(trajectories[tr]->object.cOfMass.y - vcmass[i].y,2));
		bool isntDisplacedV = (displ< max(e2.size.width*th,e2.size.height*th)) && (displ<max(e1.size.width*th,e1.size.height*th));
		bool simSizeV = sizeRatio.width<th && sizeRatio.height<th;
		bool simAngleV = difAngle<50*th;
		bool ruleMin = isntDisplacedV && simSizeV && simAngleV;

		bool isntDisplacedM = (displ< max(e2.size.width*thDecd,e2.size.height*thDecd)) && (displ<max(e1.size.width*thDecd,e1.size.height*thDecd));
		bool simSizeM = sizeRatio.width<thDecd && sizeRatio.height<thDecd;
		bool simAngleM = difAngle<60*thDecd;
		int countVote=0;
		countVote = isntDisplacedM == true ? countVote+1 : countVote;
		countVote = simSizeM == true ? countVote+1 : countVote;
		countVote = simAngleM == true ? countVote+1 : countVote;
		bool ruleVote = countVote>=2;

		return (ruleMin || ruleVote);
	}




	bool atBorder(Rect r)
	{
		bool atborder = r.x<2 || r.y<2 || (r.x+r.width)>world.width-2 || (r.y+r.height)>world.height-2;
		return atborder;
	}

	bool atBorder(Rect r, Rect ref, float dist=0.1, bool both_borders=false)
	{
		bool atborder;
		if(both_borders)
			atborder = (r.x<ref.width*dist|| r.y<ref.height*dist) && ( (r.x+r.width)>ref.width*(1-dist) || (r.y+r.height)>ref.height*(1-dist));
		else
			atborder = r.x<ref.width*dist || r.y<ref.height*dist || (r.x+r.width)>ref.width*(1-dist) || (r.y+r.height)>ref.height*(1-dist);
		return atborder;
	}
/*
	int compareVicinity(Mat & roiF, Mat & roiF_Rgb, Mat & roiExpM, int area)
	{
		float simTh=0.98;
		HessParticle p;
		float similarity =0;
		int sqrtArea;
		int patcharea=patchArea;
		sqrtArea = sqrt(area/patcharea);
		if(sqrtArea>=5)
		{
			patcharea=area/5;
			sqrtArea=5;
		}
		else if(sqrtArea<1)
		{
			sqrtArea=1;
			patcharea=1;
		}

		Mat dr, er;
		Size sd = Size(3,3);
		erode(roiExpM,er,getStructuringElement(MORPH_RECT,sd));
		dilate(roiExpM,dr,getStructuringElement(MORPH_RECT,sd));
		p.setWorld(world);

#ifdef DEBUG
//		similarity = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, roiF_Rgb, true);
#else
		similarity = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, roiF_Rgb);
#endif

		return similarity > simTh ? GHOST : VALID_OBJ;
	}
*/

	void getSpecificVicinityFeatures(Mat & roiF, Mat & roiF_Rgb, Mat & roiExpM, int area, int patcharea, vector<Point> contour, vector<float> & f3, vector<float> & f4, vector<float> & f7, vector<string> & vhf3, vector<string> & vhf4, vector<string> & vhf7, vector<int> flist,bool createHeader=false, string hbmg=string(), int bitmask=3)
	{
		//vector<float> featureValues(8);
		HessParticle p;
		p.setWorld(world);
		int sqrtArea;
		sqrtArea = round(sqrt(area/patcharea));
		int sqrtPatch=round(sqrt(patcharea));
		cv::Size grid;
		if(sqrtPatch>0)
		{
			int nw=roiF.cols/sqrtPatch;
			int nh=roiF.rows/sqrtPatch;
			int minhw=MIN(nh,nw);
			if(minhw==nw)
			{
				grid=Size(minhw+1,MAX(1,(roiF.rows*roiF.cols/(minhw+1))/patcharea));
			}
			else
			{
				grid=Size(MAX(1,(roiF.cols*roiF.rows/(minhw+1))/patcharea),minhw+1);
			}
		}
		else
			grid=Size(1,1);
		if(sqrtArea>=12)
		{
//			patcharea=area/12;
			sqrtArea=12;
		}
		else if(sqrtArea<1)
		{
			sqrtArea=1;
//			patcharea=1;
		}
		Mat dr, er;
		float v3, v4;

#ifdef CREATE_HEADER
		string hi = "i" + u.intToString(flist[4]);
		string hj = "j" + u.intToString(flist[5]);
		string hk = "k" + u.intToString(flist[6]);
		string hl = "l" + u.intToString(flist[7]);
		string hf3="f3"+hbmg+hi+hj+hk+hl;
		vhf3.push_back(hf3);
		string hf4="f4"+hbmg+hi+hj+hk+hl;
		vhf4.push_back(hf4);
		string hf7="f7"+hmg+hi+hk;
		vhf4.push_back(hf7);
#endif
//		Size sd = Size(3,3);
//
//		if(area>900)
		Size sd = Size(flist[4],flist[4]);
		erode(roiExpM,er,getStructuringElement(MORPH_RECT,sd));
		Size sd2 = Size(flist[4]+2,flist[4]+2);
		dilate(roiExpM,dr,getStructuringElement(MORPH_RECT,sd2));

		float upth=initUpth + flist[5]*incUpth;
		float difth=initDifth+flist[6]*incDifth;
		float disrate=initDisrate+flist[7]*incDisrate;

		int noZeroEr=countNonZero(er);
		if(noZeroEr<500)
		{
			er=roiExpM;
			dr=roiExpM;
		}
		if((bitmask & 1) == 1)
		{
			v3 = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, grid, contour, &f3Aver, &f4Aver, roiF_Rgb, upth, difth, disrate);
			f3.push_back(v3);
		}
		if((bitmask & 2) == 2)
		{
//			v4 = p.compareVicinityHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, contour, &f3Aver, &f4Aver, roiF_Rgb, upth, difth, disrate);
			f4.push_back(v4);
		}
		if((bitmask & 4) == 4)
		{
			float v7;
			v7 = lbspVicinitySimilarity(roiF_Rgb, roiExpM, dr, er, disrate, grid, sqrtArea);
			f7.push_back(v7);
		}
	}

    float lbspVicinitySimilarity(Mat& roiRGB, Mat& roiM, Mat& dr,	Mat& er, float disrate, Size grid, int nRowCol) {
    	Mat roiRGBc = roiRGB.clone();
		Mat dil, ero;
		Mat element = getStructuringElement(MORPH_RECT,
				Size(nRowCol * 4, nRowCol * 4));
		dilate(roiM, dil, element);
		erode(roiM, ero, element);
		Mat external, internal;
		absdiff(dil, dr, external); //absdiff(dil,roiM,external);
		absdiff(ero, er, internal); //absdiff(ero,roiM,internal);
		int featureValue = 0;
		vector < Rect > pcs;
		//u.fixedPatches(roiRGB.rows, roiRGB.cols, nRowCol, pcs, SQUARE_PATCHES);
		vector<Size> psizes;
		psizes = u.getLargerGrid(grid,Rect(0,0,roiRGB.cols,roiRGB.rows),2);
		Mat invdr=dr==0;
		vector<float> weights;
		vector<float> ftvalues;
		vector<int> setSize;
		pcs=u.findBestPatches(psizes,er,invdr,disrate,weights);
		int ncols=pcs[0].width;
		int nrows=pcs[0].height;
		int accumFeature=0;
		int count=0;

		for (uint i = 0; i < pcs.size(); i++) {
			Mat pRoiF(roiRGB, pcs[i]);
			Mat pRoiFc(roiRGBc, pcs[i]);
			Mat pRoiMI(internal, pcs[i]);
			Mat pRoiME(external, pcs[i]);
			Mat pRoiM(roiM, pcs[i]);
			int currArea = pRoiM.rows * pRoiM.cols;
			int nonZero = countNonZero(pRoiM);
			float rate = nonZero / (float) currArea;
			if (rate > disrate || rate < (1-disrate))
				continue;

			vector<KeyPoint> kpointsSobol = getRandomKeyPoints(ncols,nrows,2);
			vector<KeyPoint> vpe(kpointsSobol);
			vector<KeyPoint> vpi(kpointsSobol);
			KeyPointsFilter::runByPixelsMask(vpe,pRoiME);
			KeyPointsFilter::runByPixelsMask(vpi,pRoiMI);
			for (uint k=0; k<vpe.size(); k++) {
				vpe[k].pt.y=vpe[k].pt.y+pcs[i].y;
				vpe[k].pt.x=vpe[k].pt.x+pcs[i].x;
			}
			for (uint k=0; k<vpi.size(); k++) {
				vpi[k].pt.y=vpi[k].pt.y+pcs[i].y;
				vpi[k].pt.x=vpi[k].pt.x+pcs[i].x;
			}
			KeyPointsFilter::runByImageBorder(vpe,roiRGB.size(),2);
			KeyPointsFilter::runByImageBorder(vpi,roiRGB.size(),2);

			float ms=MAX(vpe.size(),vpi.size());
			float ratevs=(float)abs((int)(vpe.size()-vpi.size())/ms);
			if(ratevs>0.7)
				continue;

			//imwrite("/home/alex/TestData/cdw/ghost/roiRGBc.bmp",roiRGBc);
			vector<vector<ushort> > velbsp = u.getLbspSet(vpe,vpi,roiRGBc);
			vector<vector<ushort> > vilbsp = u.getLbspSet(vpi,vpe,roiRGBc);
			featureValue=u.getMaxDiffMedian(velbsp,vilbsp);
#ifdef DEBUG
			u.drawKeyPoints(vpe,roiRGBc,Scalar(255,0,150));
			u.drawKeyPoints(vpi,roiRGBc,Scalar(0,255,50));
			u.drawText(pRoiFc,SUB_SECTION,string("m="),featureValue);
			rectangle(roiRGBc,pcs[i],Scalar(0,255,50),1);
			rectangle(internal,pcs[i],Scalar(150),1);
//			for(uint i=0; i<vpe.size(); i++)
//				cout << vpe[i].pt.x << "," << vpe[i].pt.y << endl;
//			for(uint i=0; i<vilbsp.size(); i++)
//				cout << vpi[i].pt.x << "," << vpi[i].pt.y << endl;
			cout << featureValue << endl;
			imshow("pRoiF",pRoiFc);
			imshow("roiRGBc",roiRGBc);
			imshow("internal",internal);
			imshow("external",external);
			waitKey(0);
#endif
			if(featureValue>=0)
			{
				accumFeature+=featureValue;
				count++;
			}
		}
		float ftmean = -1;
		if(count>0)
			ftmean=round(accumFeature/count);
//		cout << "ftmean: " << ftmean << endl;
		return ftmean;
	}

	void getVicinityFeatures(Mat & roiF, Mat & roiF_Rgb, Mat & roiExpM, int area, int patcharea, vector<Point> contour, vector<float> & f3, vector<float> & f4, vector<float> & f7, vector<string> & vhf3, vector<string> & vhf4, vector<string> & vhf7, string hbmg=string())
	{
		vector<float> featureValues(8);
		HessParticle p;
		p.setWorld(world);
//		int sqrtArea;
//		sqrtArea = sqrt(area/patcharea);
//		if(sqrtArea>=12)
//		{
//			patcharea=area/12;
//			sqrtArea=12;
//		}
//		else if(sqrtArea<1)
//		{
//			sqrtArea=1;
//			patcharea=1;
//		}
		int sqrtPatch=round(sqrt(patcharea));
		cv::Size grid;
		if(sqrtPatch>0)
		{
			int nw=roiF.cols/sqrtPatch;
			int nh=roiF.rows/sqrtPatch;
			int minhw=MIN(nh,nw);
			if(minhw==nw)
			{
				grid=Size(minhw+1,(roiF.rows*roiF.cols/(minhw+1))/patcharea);
			}
			else
			{
				grid=Size((roiF.cols*roiF.rows/(minhw+1))/patcharea,minhw+1);
			}
		}
		else
			grid=Size(1,1);
		Mat dr, er;
		float v3, v4, v7;

		for(int i=5; i<8; i=i+2)
		{
#ifdef CREATE_HEADER
			string hi = "i" + u.intToString(i);
#endif
			Size sd = Size(i,i);
			erode(roiExpM,er,getStructuringElement(MORPH_RECT,sd));
			dilate(roiExpM,dr,getStructuringElement(MORPH_RECT,sd));
			float disrate=initDisrate;
			int noZeroEr=countNonZero(er);
			if(noZeroEr<500)
			{
				er=roiExpM;
				dr=roiExpM;
			}
			for(int l=1; l<2; l++) //era 0
			{
#ifdef CREATE_HEADER
						string hl = "l" + u.intToString(l);
#endif
				float upth=initUpth;
				for(int j=0; j<3; j++)//completo:j inicia em 0, rapido inicia em 1
				{
#ifdef CREATE_HEADER
					string hj = "j" + u.intToString(j);
#endif
					float difth=initDifth;
					for(int k=0; k<2; k++)
					{
#ifdef CREATE_HEADER
						string hk = "k" + u.intToString(k);
						string hf3="f3"+hbmg+hi+hj+hk+hl;
						vhf3.push_back(hf3);
						string hf4="f4"+hbmg+hi+hj+hk+hl;
						vhf4.push_back(hf4);
#endif
#ifdef F3
						v3 = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, grid, contour, &f3Aver, &f4Aver, roiF_Rgb, upth, difth, disrate);
						f3.push_back(v3);
//						if(!createdDescriptor)
//							maxDefault.push_back(false);
#endif
//#ifdef F4
//						v4 = p.compareVicinityHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, contour, &f3Aver, &f4Aver, roiF_Rgb, upth, difth, disrate);
//						f4.push_back(v4);
////						if(!createdDescriptor)
////							maxDefault.push_back(false);
//#endif
						difth+=incDifth;
					}
					upth+=incUpth;
				}
#ifdef CREATE_HEADER
						string hf7="f7"+hbmg+hi+hl;
						vhf7.push_back(hf7);
#endif
//				v7 = lbspVicinitySimilarity(roiF_Rgb, roiExpM, dr, er, disrate, sqrtArea);
				f7.push_back(v7);
//				if(!createdDescriptor)
//					maxDefault.push_back(true);
				disrate+=incDisrate;
			}
		}
	}

//	void getFeaturesArray(Mat & roiF, Mat & roiF_Rgb, Mat & roiExpM, int area, int patcharea, vector<float> & f7, vector<string> & vhf7, string hbmg=string())
//		{
//			vector<float> featureValues(8);
//			HessParticle p;
//			p.setWorld(world);
//			int sqrtArea;
//			sqrtArea = sqrt(area/patcharea);
//			if(sqrtArea>=12)
//			{
//				patcharea=area/12;
//				sqrtArea=12;
//			}
//			else if(sqrtArea<1)
//			{
//				sqrtArea=1;
//				patcharea=1;
//			}
//			Mat dr, er;
//			float v7;
//
//			for(int i=3; i<4; i=i+2)
//			{
//	#ifdef CREATE_HEADER
//				string hi = "i" + u.intToString(i);
//	#endif
//				Size sd = Size(i,i);
//				erode(roiExpM,er,getStructuringElement(MORPH_RECT,sd));
//				dilate(roiExpM,dr,getStructuringElement(MORPH_RECT,sd));
//
//				float disrate=initDisrate;
//				for(int l=0; l<2; l++)
//				{
//#ifdef CREATE_HEADER
//					string hl = "l" + u.intToString(l);
//					string hf3="f3"+hbmg+hi+hj+hk+hl;
//					vhf3.push_back(hf3);
//					string hf4="f4"+hbmg+hi+hj+hk+hl;
//					vhf4.push_back(hf4);
//#endif
//					v7 = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, roiF_Rgb, upth, difth, disrate, true);
//					f7.push_back(v7);
//					disrate+=incDisrate;
//				}
//			}
//		}

	float isContourBorderValid(Mat maskObj, Mat ero, int contourArea)
	{
		int pointGrid=10;
		if(contourArea<80)
		{
			pointGrid=2;
		}
		vector<Point> vp = getRandomPoints(maskObj.cols,maskObj.rows,pointGrid);
		vector<KeyPoint> vkp = u.vecPoint2Keypoint(vp,pointGrid);
		KeyPointsFilter::runByPixelsMask(vkp,ero);
		vp=u.vecKeyPoint2Point(vkp);
		if(vkp.size()==0)
		{
			vp = getRandomPoints(maskObj.cols,maskObj.rows,2);
			vkp = u.vecPoint2Keypoint(vp,pointGrid);
			KeyPointsFilter::runByPixelsMask(vkp,ero);
			vp=u.vecKeyPoint2Point(vkp);
			if(vkp.size()==0)
			{
				cv::findNonZero(ero,vp);
				//return 1;
			}
		}
		vector<Point> selectedp;
#ifdef DEBUG
		Mat outContour = Mat::zeros(maskObj.rows,maskObj.cols,CV_8UC3);
		float rate = u.evalOutContourSamples(maskObj,ero,vp,selectedp,20,outContour);
		//float rate = evalOutContourSamples(maskObj,ero,vp);//,outContour);
#else
		float rate = u.evalOutContourSamples(maskObj,ero,vp,selectedp);//,outContour);
#endif

		return rate;
	}

	float isGhostBySamples(Mat& sumth, Mat ero=Mat(), int contourArea=-1)
	{
		if(contourArea>80)
		{
		Size s1 = sumth.cols > 55 && sumth.rows > 55 ? Size(7, 7) : Size(5, 5);
		morphologyEx(sumth, openEdges, MORPH_CLOSE,getStructuringElement(MORPH_ELLIPSE, s1));
		}
		else
			openEdges=sumth;
		float isGhost = isContourBorderValid(openEdges,ero,contourArea);
		return isGhost;
	}

	bool evalGhostSVM(float rate, int weight, int matchCount, int disp)
	{
		bool isghost;
		vector<float> line{rate,(float)weight,(float)matchCount,(float)disp};
		//displaced ? line.push_back(1) : line.push_back(0);
		Mat row = u.vecToMat(line);
		float result = ratesvm.predict(row);
		isghost = result > 0;
		return isghost;
	}
	bool isGhostBayes(vector<float> sub)
	{
		bool isghost;
		u.printVector(sub,true);
		Mat row = u.vecToMat(sub);
		float result = bayes.predict(row);
		isghost = result > 0;
		return isghost;
	}

	void getFeatures(Rect cr, vector<Point> contour, int len, Mat maskc, Mat maskhull, vector<Mat> srcMats, vector<vector<int> > & ftlist, vector<float> & ftline, vector<string> & headerline,bool createHeader=false)
	{
		vector<float> ftvec;
		for(uint lst=0; lst<ftlist.size(); lst++)
		{
			ftvec = getSpecificFeatures(srcMats[0], srcMats[1], maskc, maskhull,cr,contour,len,headerline,ftlist[lst],createHeader);
			ftline.insert(ftline.end(),ftvec.begin(),ftvec.end());
		}
		createdDescriptor=true;
	}

	bool svmGhostPrediction(vector<float> & medianFeatures)
	{
		Mat row = u.vecToMat(medianFeatures);
		float result = svm.predict(row);
		bool isghost = result > 0;
		return isghost;
	}

	void detectFrameGhosts(Mat frameRGB, Mat & mask, vector<vector<int> > & ftlist)
	{
		//mydb.testQuery();
		if(countNonZero(mask)<=0)
		{
			frameNo++;
			return;
		}
//#ifdef TESTING
//ftlist.clear();
		vector<vector<int> > ftlist8;
		ftlist8.push_back(vector<int>{8,2,2});
//#endif
		u.capHoles(mask);
		Mat contrLowPass1, lowPass1, lowPassHsv1;
		Mat contrLowPass2, lowPass2, lowPassHsv2;
		Mat contrLowPass3, lowPass3, lowPassHsv3, lowPassLRG3;

		Mat filteredRGB, filteredHSV;
		bilateralFilter(frameRGB, filteredRGB, 15, 60, 20);//25, 50, 15
		//adaptiveBilateralFilter(frameRGB, filteredRGB, Size(15,15), 40, 30);
		filteredHSV = u.bgr2hsv( filteredRGB );

		//frameRGB.convertTo(contrLowPass3,-1,1.4,0);
		bilateralFilter(frameRGB, lowPass3, 15, 40, 20);//25, 50, 15
		lowPassHsv3 = u.bgr2hsv( lowPass3 );

		vector<Mat> srcMats{filteredRGB,filteredHSV,lowPass3, lowPassHsv3};

		vector<int> vAreas;
		vector<int> vLen;
		vector<Point2f> cofm;
		vector<vector<Point> > meaningfulContours;
		vector<vector<Point> > contoursHull = getContours(mask, regions, cofm, meaningfulContours, vAreas, vLen);

		//Mat cont=frameRGB.clone();
		//Mat maskCont = u.convertTo3Channels(mask);
		totalNonBlackFrames++;
		vector<string> csvlinevec;

		for (uint i=0;i<regions.size();i++)
		{
			string csvline;
			Rect cr=regions[i];
			Mat maskhull = Mat::zeros(mask.size(),CV_8U);
			Mat maskc = Mat::zeros(mask.size(),CV_8U);
			Mat roiD=Mat(maskc,cr);
			Mat roiS=Mat(mask,cr);
			roiS.copyTo(roiD);
			vector<vector<Point> > contourVec;
			contourVec.push_back(contoursHull[i]);
			drawContours( maskhull, contourVec, 0, Scalar(255), CV_FILLED);
			int cntNonZero=vAreas[i];
			//bool isAtBorder =  u.rectAtBorder(cr,world,minDiagDistance);
			bool earlyRule = cntNonZero<minArea;
			if(earlyRule)
			{
				cout << "cntNonZero<minArea" << endl;
				continue;
			}

			vector<float> ftvectotal;
			vector<float> ft8;
			vector<string> vhtotal;
			vector<string> vh8;
			getFeatures(cr,meaningfulContours[i],vLen[i],maskc,maskhull,srcMats,ftlist,ftvectotal,vhtotal,i==0);
			getFeatures(cr,meaningfulContours[i],vLen[i],maskc,maskhull,srcMats,ftlist8,ft8,vh8,i==0);

			bool isghost=false;
			u.zeroNegatives(ftvectotal);
#ifdef EVAL_SVM
			if(vAreas[i]>200)
			{
				bool svmPred = svmGhostPrediction(ftvectotal);
				isghost = ( svmPred && ftvectotal[2]>0.1) || ft8[0];
			}
			else
				isghost = ftvectotal[2]>0.5 || ft8[0];
#endif
#ifdef DEBUG
//		imshow( "filteredRGB", filteredRGB );
//			if(isghost)
//				waitKey(0);
#endif
#ifdef CREATE_HEADER
			if(i==0 && !createdHeader)
			{
				createdHeader=true;
				u.appendToFile(vhtotal,outpath+"/header.csv",true);
			}
#endif
#ifdef TRAINING
			csvline=outputResultBlob(meaningfulContours[i],maskc,filteredRGB,outpath,frameNo,cr,isghost,i,ftvectotal,true);
#else
			csvline=outputResultBlob(meaningfulContours[i],maskc,filteredRGB,outpath,frameNo,cr,isghost,1000*frameNo+i,ftvectotal,false);
#endif
			csvlinevec.push_back(csvline);
			totalGhosts = isghost ? totalGhosts+1 : totalGhosts;
			totalNoGhosts = isghost ? totalNoGhosts : totalNoGhosts+1;
			totalBlobsAnalyzed++;
		}
		u.appendToFile(csvlinevec,outpath+"/features.csv");

		frameNo++;
	}

	vector<vector<float> > getFrameFeatures(Mat frameRGB, Mat & mask, vector<vector<int> > & ftlist, vector<int> & areas, vector<vector<Point> > & analyzedContours, vector<Rect> & rects)
	{
		vector<vector<float> > vvfeatures;
		//mydb.testQuery();
		if(countNonZero(mask)<=0)
		{
			frameNo++;
			return vvfeatures;
		}
//#ifdef TESTING
//ftlist.clear();
//		vector<vector<int> > ftlist8;
//		ftlist8.push_back(vector<int>{8,2,2});
//#endif
		u.capHoles(mask);

		Mat filteredRGB, filteredHSV;
		filteredRGB = frameRGB;
		//bilateralFilter(frameRGB, filteredRGB, 15, 60, 20);//25, 50, 15
		//adaptiveBilateralFilter(frameRGB, filteredRGB, Size(15,15), 40, 30);
		filteredHSV = u.bgr2hsv( filteredRGB );

		//frameRGB.convertTo(contrLowPass3,-1,1.4,0);
//		bilateralFilter(frameRGB, lowPass3, 15, 40, 20);//25, 50, 15
//		lowPassHsv3 = u.bgr2hsv( lowPass3 );

		vector<Mat> srcMats{filteredRGB,filteredHSV};//,lowPass3, lowPassHsv3};

		vector<int> vAreas;
		vector<int> vLen;
		vector<Point2f> cofm;
		vector<vector<Point> > meaningfulContours;
		vector<vector<Point> > contoursHull = getContours(mask, regions, cofm, meaningfulContours, vAreas, vLen);

		//Mat cont=frameRGB.clone();
		//Mat maskCont = u.convertTo3Channels(mask);
		totalNonBlackFrames++;
		vector<string> csvlinevec;
		//areas = vAreas;

		for (uint i=0;i<regions.size();i++)
		{
			string csvline;
			Rect cr=regions[i];
			Mat maskhull = Mat::zeros(mask.size(),CV_8U);
			Mat maskc = Mat::zeros(mask.size(),CV_8U);
			Mat roiD=Mat(maskc,cr);
			Mat roiS=Mat(mask,cr);
			roiS.copyTo(roiD);
			vector<vector<Point> > contourVec;
			contourVec.push_back(contoursHull[i]);
			drawContours( maskhull, contourVec, 0, Scalar(255), CV_FILLED);
			int cntNonZero=vAreas[i];
			//bool isAtBorder =  u.rectAtBorder(cr,world,minDiagDistance);
			bool earlyRule = cntNonZero<minArea;
			if(earlyRule)
			{
				cout << "cntNonZero<minArea" << endl;
				continue;
			}

			vector<float> ftvectotal;
			vector<string> vhtotal;
			getFeatures(cr,meaningfulContours[i],vLen[i],maskc,maskhull,srcMats,ftlist,ftvectotal,vhtotal,i==0);
			bool isghost=false;
			//u.zeroNegatives(ftvectotal);
			vvfeatures.push_back(ftvectotal);
			areas.push_back(vAreas[i]);
			analyzedContours.push_back(meaningfulContours[i]);
			rects.push_back(cr);
		}
		return vvfeatures;
	}

	vector<float> getSpecificFeatures(Mat frameRGB, Mat & frameHSV, Mat & maskc, Mat & hull, Rect & region, vector<Point> contour, int len, vector<string> & header, vector<int> flist,bool createHeader=false)
	{
		vector<float> featureValues;
		vector<float> f10,f20,f30,f40,f50,f60,f70,f80;
		vector<string> vhf10,vhf20,vhf30,vhf40,vhf50,vhf60,vhf70,vhf80;
		vector<float> tmp30,tmp40,tmp70;
		vector<string> tmph30,tmph40,tmph70;
		vector<int> toRemove;
		Rect alias = region;
		float resizeFactor=0;
		if(flist[2]>=0)
			resizeFactor=initResizeFactor + incResizeFactor*flist[2];

//		if(flist[0]==6)
//			resizeFactor*=8;

		world = Rect(0,0,frameRGB.cols,frameRGB.rows);
		Mat hsvf = frameHSV;
		Mat mroi_before(maskc,alias);
		int countContours;
		Rect largestRect;
		vector<Point> vp;
		Mat mroi_result = u.getLargestContour(mroi_before, largestRect, vp, &countContours);
		largestRect.x+=alias.x;
		largestRect.y+=alias.y;
		alias=largestRect;

		Rect exp =expandRect(alias, resizeFactor);
		exp = exp & world;
		for(uint i=0; i<contour.size(); i++)
		{
			contour[i].x=contour[i].x-exp.x;
			contour[i].y=contour[i].y-exp.y;
		}
		int resizedArea=exp.width*exp.height;
		Mat roiM = mroi_result;
		Mat maskclone = Mat::zeros(maskc.size(),CV_8U);
		Mat roiCopyM(maskclone,alias);
		mroi_result.copyTo(roiCopyM);
		Mat roiExpM = Mat(maskclone,exp);
		Mat roiF_Rgb = Mat(frameRGB,exp);
		Mat roiF = Mat(hsvf,exp);
		Mat roiExpHull = Mat(hull,exp);
		Mat roiRGB;

		float weight=0;
		int currpatcharea=0;
		if(flist[3]>=0)
		{
			weight=initWeight+flist[3]*incWeight;
			currpatcharea=initCurrpatcharea+flist[3]*incCurrpatcharea;
		}

		string hb = "b" + u.intToString(flist[1]);
		string hm = "m" + u.intToString(flist[2]);
		string hg = "g" + u.intToString(flist[3]);
		string hbmg = hb+hm+hg;

		float borderRate;
		float f1, v20, v60;
		string hf1,hf2,hf3,hf4,hf5,hf6,hf7;
		string hi;
		Mat er,dr;
		Size sd;
		HessParticle p;
		Mat mopen; //TODO:Melhorar esta regra
		int origArea;
		int eroArea;
		bool hasCB;

		switch(flist[0])
		{
		case 1:
			p.setWorld(world);
			f1 = p.compareIntExtHistograms(roiF, roiExpM);
			f10.push_back(f1);
			featureValues.insert(featureValues.end(),f10.begin(),f10.end());
#ifdef CREATE_HEADER
			hf1="f1"+hb+hm;
			vhf10.push_back(hf1);
			header.insert(header.end(),vhf10.begin(),vhf10.end());
#endif
			break;
		case 2:
#ifdef CREATE_HEADER
			hf2="f2"+hb+hm+hg;
			vhf20.push_back(hf2);
			header.insert(header.end(),vhf20.begin(),vhf20.end());
#endif
			sd = Size(flist[3],flist[3]);
			erode(roiExpM,er,getStructuringElement(MORPH_RECT,sd));
			dilate(roiExpM,dr,getStructuringElement(MORPH_RECT,sd));
			p.setWorld(world);
			v20 = p.compareVicinityHistograms(roiF, roiExpM, dr, er, roiF_Rgb);
			f20.push_back(v20);
			featureValues.insert(featureValues.end(),f20.begin(),f20.end());
			break;
		case 3:
			getSpecificVicinityFeatures(roiF, roiF_Rgb, roiExpM, resizedArea, currpatcharea, contour, tmp30,tmp40,tmp70,tmph30,tmph40,tmph70,flist, createHeader, hbmg,1);
			f30=tmp30;
			vhf30=tmph30;
#ifdef CREATE_HEADER
			header.insert(header.end(),vhf30.begin(),vhf30.end());
#endif
			featureValues.insert(featureValues.end(),f30.begin(),f30.end());
			break;
		case 4:
			getSpecificVicinityFeatures(roiF, roiF_Rgb, roiExpM, resizedArea, currpatcharea, contour, tmp30,tmp40,tmp70,tmph30,tmph40,tmph70,flist, createHeader, hbmg,2);
			f40=tmp40;
			vhf40=tmph40;
#ifdef CREATE_HEADER
			header.insert(header.end(),vhf40.begin(),vhf40.end());
#endif
			featureValues.insert(featureValues.end(),f40.begin(),f40.end());
			break;
		case 5:
#ifdef CREATE_HEADER
			hi = "i" + u.intToString(flist[4]);
			hf5="f5"+hb+hm+hg+hi;
			vhf50.push_back(hf5);
			header.insert(header.end(),vhf50.begin(),vhf50.end());
#endif
			cv::morphologyEx(roiM,mopen,MORPH_OPEN,getStructuringElement(MORPH_RECT, Size(5,5)));
			origArea=countNonZero(roiM);
			eroArea=countNonZero(mopen);
			if(origArea>3*eroArea)
			{
				f50.push_back(-1);
				featureValues.insert(featureValues.end(),f50.begin(),f50.end());
				break;
			}
			roiRGB = roiF_Rgb.clone();
			borderRate = 1-samplesBorderWithHull(roiRGB, roiExpM, roiExpHull, contour, origArea);//, roiExpDif);
//				if(borderRate<0)
//					cerr<< "borderRate<0" << endl;
//				cout << "borderRate=" <<borderRate << endl;

			f50.push_back(borderRate);
			featureValues.insert(featureValues.end(),f50.begin(),f50.end());
			break;
		case 6:
#ifdef CREATE_HEADER
			hf6="f6"+hb+hm+hg;
			vhf60.push_back(hf6);
			header.insert(header.end(),vhf60.begin(),vhf60.end());
#endif
			//sd=Size(3,3);
			v60=applyGrabCut(roiF_Rgb,roiExpM,flist[2]);
			f60.push_back(v60);
			featureValues.insert(featureValues.end(),f60.begin(),f60.end());
			break;
		case 7:
#ifdef CREATE_HEADER
			hf7="f7"+hb+hm+hg;
			vhf70.push_back(hf7);
			header.insert(header.end(),vhf70.begin(),vhf70.end());
#endif
			//sd=Size(3,3);
			getSpecificVicinityFeatures(roiF, roiF_Rgb, roiExpM, resizedArea, currpatcharea, contour, tmp30,tmp40,tmp70,tmph30,tmph40,tmph70,flist, createHeader, hbmg,4);
			f70=tmp70;
			vhf70=tmph70;
			featureValues.insert(featureValues.end(),f70.begin(),f70.end());
			break;
		case 8:
			hasCB = hasCrossingBorders(roiF_Rgb,roiExpM,contour, len);
			if(hasCB)
				f80.push_back(1);
			else
				f80.push_back(0);

			featureValues.insert(featureValues.end(),f80.begin(),f80.end());
			break;
		default:
			break;
		}
		return featureValues;
	}

	vector<float> getFeature4(Mat frameRGB, Mat & frameHSV, Mat & maskc, Mat & hull, Rect & region, vector<Point> contour, vector<string> & header, vector<int> flist,bool createHeader=false)
	{
		vector<float> featureValues;
		vector<float> f40;
		vector<string> vhf40;
		vector<float> tmp40;
		vector<string> tmph40;

		Mat frameblur;
		vector<int> toRemove;
		Rect alias = region;
		float resizeFactor=0.9;

		frameblur = frameRGB;
		world = Rect(0,0,frameRGB.cols,frameRGB.rows);
		Mat hsvf = frameHSV;
		Mat mroi_before(maskc,alias);
		int countContours;
		Rect largestRect;
		vector<Point> vp;
		Mat mroi_result = u.getLargestContour(mroi_before, largestRect, vp, &countContours);
		largestRect.x+=alias.x;
		largestRect.y+=alias.y;
		alias=largestRect;
		Rect exp =expandRect(alias, resizeFactor);
		exp = exp & world;
		for(uint i=0; i<contour.size(); i++)
		{
			contour[i].x=contour[i].x-exp.x;
			contour[i].y=contour[i].y-exp.y;
		}
		int resizedArea=exp.width*exp.height;
		//Mat maskContRoi = Mat(maskCont,exp);
		Mat roiM = mroi_result;
		Mat maskclone = Mat::zeros(maskc.size(),CV_8U);
		Mat roiCopyM(maskclone,alias);
		mroi_result.copyTo(roiCopyM);
		Mat roiExpM = Mat(maskclone,exp);
		Mat roiF_Rgb = Mat(frameRGB,exp);
		Mat roiF = Mat(hsvf,exp);
		Mat roiExpHull = Mat(hull,exp);
		Mat roiRGB;

		int currpatcharea=0;
		currpatcharea=initCurrpatcharea+flist[3]*incCurrpatcharea;

		string hb = "b" + u.intToString(flist[1]);
		string hm = "m" + u.intToString(flist[2]);
		string hg = "g" + u.intToString(flist[3]);
		string hbmg = hb+hm+hg;

		float borderRate;
		float f1, v20, v60;
		string hf1,hf2,hf3,hf4,hf5,hf6,hf7;
		string hi;
		Mat er,dr;
		Size sd;
		HessParticle p;
		Mat mopen; //TODO:Melhorar esta regra
		int origArea;
		int eroArea;
		getSpecificVicinityFeatures(roiF, roiF_Rgb, roiExpM, resizedArea, currpatcharea, contour, tmp40,tmp40,tmp40,tmph40,tmph40,tmph40,flist, createHeader, hbmg,2);
		f40=tmp40;
		vhf40=tmph40;
#ifdef CREATE_HEADER
			header.insert(header.end(),vhf40.begin(),vhf40.end());
#endif
		featureValues.insert(featureValues.end(),f40.begin(),f40.end());
		return featureValues;
	}




	float getEdgesStdDev(Mat sob8b, Rect & region)
	{
		float resizeFactor=0.4;
		Rect exp = expandRect(region, resizeFactor);
		exp = exp & world;
		Mat sob8bExpRgb = Mat(sob8b,exp);
		//Mat sobedges = roiExpRgb.clone();
		//u.applySobel(sobedges);
		//Mat sob8b;
		//cvtColor(sobedges, sob8b, CV_RGB2GRAY);
		Scalar mean;
		Scalar std;
		meanStdDev(sob8bExpRgb,mean,std);
		return (float)std[0];
	}

	vector<float> getFeature6(Mat frameRGB, Mat & frameHSV, Mat & maskc, Mat & hull, Rect & region, Mat & rgbCont, vector<string> & header, string hb=string(),bool createHeader=false)
	{
		//vector<float> featureValues;
		vector<float> f60;
		vector<string> vhf60;

		Mat frameblur;
		vector<int> toRemove;
		Rect alias = region;
		float resizeFactor=0.4;
		frameblur = frameRGB;

		world = Rect(0,0,frameRGB.cols,frameRGB.rows);
		Mat hsvf = frameHSV;
		Mat mroi_before(maskc,alias);
		int countContours;
		Rect largestRect;
		vector<Point> vp;
		Mat mroi_result = u.getLargestContour(mroi_before, largestRect, vp, &countContours);
		largestRect.x+=alias.x;
		largestRect.y+=alias.y;
		alias=largestRect;

		Rect exp =expandRect(alias, resizeFactor);
		exp = exp & world;
		//int resizedArea=exp.width*exp.height;
		//Mat maskContRoi = Mat(maskCont,exp);
		Mat roiM = mroi_result;
		Mat maskclone = Mat::zeros(maskc.size(),CV_8U);
		Mat roiCopyM(maskclone,alias);
		mroi_result.copyTo(roiCopyM);
		Mat roiExpM = Mat(maskclone,exp);
		Mat roiF_Rgb = Mat(rgbCont,exp);
		//Mat roiF = Mat(hsvf,exp);
		//roiF_Rgb.convertTo(roiF,-1,2,0);
		//roiF=u.bgr2hsv(roiF);

#ifdef CREATE_HEADER
		string hg,hf6;
		if(createHeader)
		{
			hg = "g0";
			hf6="f6"+hb+hg;
			vhf60.push_back(hf6);
			hg = "g7";
			hf6="f6"+hb+hg;
			vhf60.push_back(hf6);
		}
#endif
		float v60=applyGrabCut(roiF_Rgb,roiExpM,0);
		f60.push_back(v60);
		v60=applyGrabCut(roiF_Rgb,roiExpM,7);
		f60.push_back(v60);

		//featureValues.insert(featureValues.end(),f60.begin(),f60.end());
		header.insert(header.end(),vhf60.begin(),vhf60.end());

		return f60;
	}

	Mat getFrameDiff(Mat & currFrame)
	{
		if(difFrame!=frameNo)
		{
			lastFrame[3]=lastFrame[2];
			lastFrame[2]=lastFrame[1];
			lastFrame[1]=lastFrame[0];
			lastFrame[0]=currFrame.clone();
			difFrame=frameNo;
			if(lastFrame[3].empty())
			{
				frameDiff = Mat::zeros(currFrame.size(),CV_8U);
			}
			else
			{
				Mat dif;
				cv::absdiff(currFrame,lastFrame[3],dif);
				cvtColor(dif, frameDiff, CV_BGR2GRAY);
				u.applySobel(frameDiff);
				threshold(frameDiff,frameDiff,115,255,CV_THRESH_BINARY);
			}
		}
		return frameDiff;
	}

//	vector<int> removeGhostRegionWithHull(Mat frameRGB, Mat & frameHSV, Mat & maskc, Mat & hull, int nframe, int ID, bool smallStd, Rect & region)
//	{
//		vector<int> codes(2);
//		int validCode;
//		int histCompareCode;
//		Mat frameblur;
//		vector<int> toRemove;
//		Rect alias = region;
//		int area = region.width*region.height;
//		float resizeFactor=0.4;
//		frameblur = frameRGB;
//		Mat srcClone = frameRGB.clone();
//		world = Rect(0,0,frameRGB.cols,frameRGB.rows);
//		Mat hsvf = frameHSV;
//		Mat mroi_before(maskc,alias);
//		int countContours;
//		Rect largestRect;
//		vector<Point> vp;
//		Mat mroi_result = u.getLargestContour(mroi_before, largestRect, vp, &countContours);
//		largestRect.x+=alias.x;
//		largestRect.y+=alias.y;
//		//Mat roih(hsv,largestRect);
//		alias=largestRect;
//		Mat fdiff = getFrameDiff(frameRGB);
//		Rect exp =expandRect(alias, resizeFactor);
//		exp = exp & world;
//		//Mat maskContRoi = Mat(maskCont,exp);
//		//Mat roiM = Mat(maskc,alias);
//		Mat roiM = mroi_result;
//		Mat maskclone = Mat::zeros(maskc.size(),CV_8U);
//		Mat roiCopyM(maskclone,alias);
//		mroi_result.copyTo(roiCopyM);
//		Mat roiExpM = Mat(maskclone,exp);
//		Mat roiDiff = Mat(fdiff,exp);
//		Mat roiF_Rgb = Mat(frameRGB,exp);
//		Mat roiF;// = Mat(hsvf,exp);
//		roiF_Rgb.convertTo(roiF,-1,2,0);
//		roiF=u.bgr2hsv(roiF);
//		Mat roiExpDif = Mat::zeros(roiF.size(),CV_8U);
//		roiDiff.copyTo(roiExpDif,roiExpM);
//		Mat roiRGB;
//		roiRGB = Mat(frameblur, exp).clone();
//
//		histCompareCode=compareVicinity(roiF, roiF_Rgb, roiExpM, area);
//		codes[1]=histCompareCode;
//
//		Mat mopen; //TODO:Melhorar esta regra
//		cv::morphologyEx(roiM,mopen,MORPH_OPEN,getStructuringElement(MORPH_RECT, Size(5,5)));
//		int origArea=countNonZero(roiM);
//		int eroArea=countNonZero(mopen);
//		if(origArea>3*eroArea)
//		{
//			ghost = roiM.clone();
//			validCode=THIN_AREA;
//			codes[0]=VALID_OBJ;
//			return codes;
//		}
//
//		Mat roiExpHull = Mat(hull,exp);
//		//		imshow("hull", hull);
//		//		waitKey(0);
//		validCode = isObjectValidWithHull(roiRGB, roiExpM, roiExpHull, smallStd, roiExpDif);
//
//		if(validCode<=0)
//		{
//#ifdef WRITE_GHOSTS
//			Mat maskth;
//			string imname = string(outpath);
//			string oename = string(outpath);
//			string edname = string(outpath);
//			string roimname = string(outpath);
//			imname.append("/RGB");
//			oename.append("/OpenEdges");
//			edname.append("/Contour");
//			roimname.append("/roiM");
//#endif
//			ghost = roiM.clone();
//		}
//		else
//			ghost = Mat::zeros(roiM.size(),CV_8U);
//
//
//		codes[0]=validCode > 0 ? VALID_OBJ : GHOST;
//		return codes;
//	}

	int cvuiIsRegionGhost(Mat frameRGB, Mat & maskc, Mat & hull, int nframe, int ID, bool smallStd, Rect & region)
	{
		vector<int> codes(1);
		int validCode;
		//int histCompareCode;
		Mat frameblur;
		vector<int> toRemove;
		Rect alias = region;
		//int area = region.width*region.height;
		float resizeFactor=0.4;
		frameblur = frameRGB;
		Mat srcClone = frameRGB.clone();
		Mat fdiff = getFrameDiff(frameRGB);
		world = Rect(0,0,frameRGB.cols,frameRGB.rows);
		Mat mroi_before(maskc,alias);
		int countContours;
		Rect largestRect;
		vector<Point> vp;
		Mat mroi_result = u.getLargestContour(mroi_before, largestRect, vp, &countContours);
		largestRect.x+=alias.x;
		largestRect.y+=alias.y;
		alias=largestRect;
		Rect exp =expandRect(alias, resizeFactor);
		exp = exp & world;
		//Mat maskContRoi = Mat(maskCont,exp);
		Mat roiM = mroi_result;
		Mat maskclone = Mat::zeros(maskc.size(),CV_8U);
		Mat roiCopyM(maskclone,alias);
		mroi_result.copyTo(roiCopyM);
		Mat roiExpM = Mat(maskclone,exp);
		Mat roiDiff = Mat(fdiff,exp);
		Mat roiF_Rgb = Mat(frameRGB,exp);
		Mat roiExpDif = Mat::zeros(roiDiff.size(),CV_8U);
		roiDiff.copyTo(roiExpDif,roiExpM);
		Mat roiRGB;
		roiRGB = Mat(frameblur, exp).clone();

		Mat mopen; //TODO:Melhorar esta regra
		cv::morphologyEx(roiM,mopen,MORPH_OPEN,getStructuringElement(MORPH_RECT, Size(5,5)));
		int origArea=countNonZero(roiM);
		int eroArea=countNonZero(mopen);
		if(origArea>3*eroArea)
		{
			return -1;
		}

		Mat roiExpHull = Mat(hull,exp);
		//		imshow("hull", hull);
		//		waitKey(0);
		validCode = isObjectValidWithHull(roiRGB, roiExpM, roiExpHull, smallStd, roiExpDif);

		if(validCode<=0)
		{
#ifdef WRITE_GHOSTS
			Mat maskth;
			string imname = string(outpath);
			string oename = string(outpath);
			string edname = string(outpath);
			string roimname = string(outpath);
			imname.append("/RGB");
			oename.append("/OpenEdges");
			edname.append("/Contour");
			roimname.append("/roiM");
#endif
			ghost = roiM.clone();
		}
		else
			ghost = Mat::zeros(roiM.size(),CV_8U);

		bool isghost =validCode > 0 ? VALID_OBJ : GHOST;
		return isghost;
	}


	/*
	 * O src1 contem os elementos que serao considerados para a ordenacao.
	 * O src2 seguirá a ordenacao estabelecida no src1
	 */
	void sortTwoVectors(vector<int> & src1, vector<int> & src2)
	{
		// 			printVector(src1);
		// 			printVector(src2);
		vector <int> p = sort_permutation(src1);
		src1 = apply_permutation(src1, p);
		src2 = apply_permutation(src2, p);
		//			printVector(src1);
		//			printVector(src2);
	}

	bool hasCrossingBorders(Mat& src, Mat& mask, vector<Point> contour, int len)
	{
		Mat sumth,edges8b,openned;
		Mat sobedges = src.clone();
		u.applySobel(sobedges);
		cvtColor(sobedges, edges8b, CV_BGR2GRAY);
		Scalar MED = mean(edges8b);
		Scalar MAD = abs(mean(MED - edges8b).val[0]);
		int sumthrsh;
//		Scalar means,std;
		sumthrsh = (int) MIN(100,MAX(10,(MED.val[0] + 4 * 1.4826 * (MAD.val[0]))));
//		meanStdDev(sobedges,means,std);
//		sumthrsh = (int) MAX(30,means[0]*1.1);
		threshold(edges8b, sumth, sumthrsh, 255, THRESH_BINARY);

//		Size s1 = Size(3, 3);
//		morphologyEx(sumth, openned, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, s1));
//		absdiff(sumth,openned,sumth);

		vector<vector<Point> >  thin = getLines(sumth);
		Mat drawThin=Mat::zeros(src.size(),CV_8U);
		drawContours(drawThin,thin,-1,Scalar(255));
		Mat inter;
		bitwise_and(drawThin, mask, inter);
		Mat diff;
		absdiff(drawThin,mask,diff);
		int dlen=0;
		getLines(diff,&dlen);
		int slen=0;
		getLines(inter,&slen);
		bool result=slen*6>len && dlen*8>len;
#ifdef DEBUG
//		if(result)
//			cout << thin.size() << ", slen=" << slen << ", len=" << len << endl;
//
//		vector<vector<Point> > contourVec;
//		contourVec.push_back(contour);
//		Mat draw = sumth.clone();
//		drawContours( draw, contourVec, 0, Scalar(255));
//		imshow("sobedges", sobedges);
//		imshow("edges8b", edges8b);
//		imshow("sumth", sumth);
//		imshow("draw", draw);
		//imshow("openned", openned);
//		imshow("thin", drawThin);
//		imshow("inter", inter);
		//waitKey(0);
#endif
		return result;
	}

	float samplesBorderWithHull(Mat& src, Mat& mask, Mat& hull, vector<Point> contour, int contourArea)
	{
		Mat dil, ero;
		Mat sob2;
		Mat sobedges = src.clone();
		Mat susanedges = src.clone();
		Mat sumEdges;
		Mat sum8b;
		Mat sumth;

		Size sd = Size(5, 5);
		dilate(hull, dil, getStructuringElement(MORPH_ELLIPSE, sd));
		Size se = Size(5, 5);
		erode(mask, ero, getStructuringElement(MORPH_ELLIPSE, se));
		int eroArea=countNonZero(ero);
		if(eroArea==0)
		{
			sd = Size(3, 3);
			dilate(hull, dil, getStructuringElement(MORPH_ELLIPSE, sd));
			se = Size(3, 3);
			erode(mask, ero, getStructuringElement(MORPH_ELLIPSE, se));
			eroArea=countNonZero(ero);
			if(eroArea==0)
				return 1;
		}
		Mat coroa;
		coroa = dil & (ero == 0);
		Mat sobCoroa, canCoroa;
//#ifdef SUSAN
		//Mat src8b;
//		cvtColor(sobedges, sobedges, CV_BGR2GRAY);
		u.susanEdges3C(susanedges,16);
		//sumth = sobedges & coroa;
//#else
		u.applySobel(sobedges);
		cvtColor(sobedges, sobedges, CV_BGR2GRAY);

//		vector<Mat> planes, vedges(3);
//		vector<>
//		split(sobedges, planes);


		susanedges = susanedges+sobedges;
//#endif
//		add(sobedges,susanedges,sobedges);

		//sum8b = sobedges & coroa;
		Scalar means;
		Scalar std;
		meanStdDev(sobedges,means,std);
		int sumthrsh;
		//sumthrsh = (int) (MED.val[0] + weight * 1.4826 * (MAD.val[0]+MAX(2,std[0]/7)));
		int th2 = (int) MAX(110,means[0]);
		Mat masklines;
		threshold(susanedges, masklines, th2, 255, THRESH_BINARY);

		vector<vector<Point> >  thin = getLines(masklines,NULL,0.1,true);
		if(thin.size()>0)
		{
			drawContours(susanedges,thin,-1,Scalar(0));
#ifdef DEBUG
			Mat thinline = Mat::zeros(sobedges.size(),CV_8U);
			drawContours(thinline,thin,-1,Scalar(255));
//			imshow("thinline",thinline);
//			imshow("masklines",masklines);
#endif
		}
		sum8b = susanedges & coroa;
		sumthrsh = (int) MAX(10,means[0]+0.25*std[0]);
		threshold(sum8b, sumth, sumthrsh, 255, THRESH_BINARY);
		//meanStdDev(sobedges,means,std);

#ifdef DEBUG
//		imshow("sum8b",sum8b);
#endif

		vector<KeyPoint> vkp = u.vecPoint2Keypoint(contour);
		KeyPointsFilter::runByImageBorder(vkp,src.size(),2);
		vector<Point> vpNotAtBorder = u.vecKeyPoint2Point(vkp);
//		cout << "sizes:" << contour.size() << "," << vpNotAtBorder.size() << endl;
		vector<Point> vpAtBorder = u.getComplementSet(contour,vpNotAtBorder);
		if(vpAtBorder.size()>0)
		{
			vector<vector<Point> > vcs;
			vcs.push_back(vpAtBorder);
			cv::polylines(sumth,vcs,0,Scalar(255));
		}
#ifdef DEBUG
		vector<vector<Point> > vcsdbg;
		vcsdbg.push_back(contour);

		drawContours(src,vcsdbg,0,Scalar(0,0,255),1);
		imshow("Edges",susanedges);
		imshow("Source",src);
		imshow("Selected Edges",sumth);
		imshow("Neighborhood",coroa);
		imshow("Mask",mask);
		waitKey(10);
#endif
//		imwrite("/media/alex/3afb1861-a12a-472d-9dda-ac2b0bdfb6c6/write_test/cdw/ghost/sum8b.bmp",sum8b);
//		imwrite("/media/alex/3afb1861-a12a-472d-9dda-ac2b0bdfb6c6/write_test/cdw/ghost/sumth.bmp",sumth);
//		imshow("sClone",sClone);
		float result;
		result = isGhostBySamples(sumth,ero,contourArea);

		return result;
	}


	int isObjectValidWithHull(Mat& src, Mat& mask, Mat& hull, bool lowStd, Mat tdiff=Mat())
	{
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
		if(lowStd)
			sumthrsh = (int) (((MED.val[0] + 1 * 1.4826 * MAD.val[0])));
		else
			sumthrsh = (int) (((MED.val[0] + 2 * 1.4826 * MAD.val[0])));

		threshold(sum8b, sumth, sumthrsh, 255, THRESH_BINARY);
		//		imshow("mask.png", mask);
		//		imwrite("sobedges.png", sobedges);
		//				imwrite("ero.png", ero);

		int result = 1; //false;
		result = findObjects(sumth, drawCont);
		if (result > 0)
		{
			return result;
		}
		Mat diffEdges = tdiff;
		sumth = sumth | (diffEdges & coroa);
		result = findObjects(sumth, drawCont);
		return result;
	}

	float applyGrabCut(Mat & srcFrame, Mat srcMask, int kernelSize)
	{
		Mat destMask;
		Mat bgModel,fgModel;
		Mat foreground, tmp;
		Mat dr3, dr5;
		int sqrtArea;
		int area=countNonZero(srcMask);

		sqrtArea = sqrt(area/minArea);
		if(sqrtArea>=12)
			sqrtArea=12;
		else if(sqrtArea<1)
			sqrtArea=1;

		int ksize;
		if(kernelSize==0)
			ksize=round((1+(2*sqrtArea+1))/1.2);
		else
			ksize=kernelSize;

		destMask = buildMask(srcMask, ksize);
		dilate(srcMask,dr3,getStructuringElement(MORPH_RECT,Size(3,3)));
		dilate(srcMask,dr5,getStructuringElement(MORPH_RECT,Size(5,5)));
		Mat crown;
		absdiff(dr3,dr5,crown);
		if(destMask.cols==0)
			return 0;

		try {
			grabCut(srcFrame,destMask,Rect(),bgModel, fgModel,1,GC_INIT_WITH_MASK);
		}
		catch (cv::Exception & e) 	{ 	return 0; 	}
		catch (std::exception & e)	{ 	return 0;	}

		compare(destMask,GC_PR_FGD,foreground,cv::CMP_EQ);
		compare(destMask,GC_FGD,tmp,cv::CMP_EQ);
		Mat combined=foreground+tmp;
		Mat selected = combined & crown;
#ifdef DEBUG
//		imshow("combined",combined);
//		imshow("srcMask",srcMask);
//		imshow("crown",crown);
//		imshow("selected",selected);
//		waitKey(0);
#endif
		int countSelected=countNonZero(selected);
		int countCrown=countNonZero(crown);
		float rate=0;
		if(countSelected>0)
			rate = countSelected/(float)countCrown;
		if(rate>1)
			rate=1;

		return rate;
	}

	void writeGhost(vector<vector<Point> > contours, Mat & ghostmask, Mat & img, string path, int framenum, string name, bool isghost, string titletext=string(), string sectiontext=string())
	{
		if(img.cols==0 || ghostmask.cols==0)
			return;
		string outname = string(path)+name;
		string inname = string(path)+name;
		//outname.append("/ghost/");
		Mat imgclone=img.clone();
		inname.append("src");
		//if(ghostmask.channels()==3)
		drawContours( imgclone, contours, 0, Scalar(0,0,255), 1 );

#ifndef REPORT_RESULT
		drawContours( ghostmask, contours, 0, Scalar(255), CV_FILLED );
		u.drawText(ghostmask,SUB_TITLE,titletext);
		u.drawText(imgclone,SUB_TITLE,titletext);
		u.drawText(ghostmask,SECTION,sectiontext);
		u.drawText(imgclone,SECTION,sectiontext);
#else
		if(isghost)
		{
			drawContours( ghostmask, contours, 0, Scalar(0), CV_FILLED );
#ifdef DEBUG
//			imshow("imgghost",imgclone);
//			imshow("maskghost",ghostmask);
//			waitKey(0);
#endif
		}
#endif
		if(ghostmask.empty() || imgclone.empty())
			cerr << "ghostmask.empty() || imgclone.empty()" << endl;
		u.writePNG(outname, ghostmask);
		if(isghost)
			u.writePNG(inname, imgclone);
	}

	string outputResultBlob(vector<Point> contour, Mat & ghostmask, Mat img, string path, int nframe, Rect rec, bool isGhost, int ID, vector<float> ftvec=vector<float>(), bool bypassClass=false)
	{
		vector<vector<Point> > contours;
		contours.push_back(contour);
		string recstr=rectToString(rec);
		string framestr = u.intToString(nframe);
		string filename=framestr+"_"+recstr;
		string ghostpath=outpath+"/ghost/";
		string noghostpath=outpath+"/noghost/";
		string csvline = framestr + "," + recstr;
		//csvline=csvline+","+segmentation;
		//csvline=csvline+","+testcase;
		//csvline=atBorder ? csvline+",1" : csvline+",0";
		//csvline=csvline+","+u.intToString(area);
		//bool repeated=isRecorded(nframe,recstr,segmentation,testcase);
		csvline=csvline+u.floatToString(ftvec);

		if(bypassClass)
		{
			csvline=csvline+",I";
		}
		else
		{
#ifdef OUTPUTALL
			if(isGhost)
			{
				csvline=csvline+",G";
				writeGhost(contours,ghostmask,img,ghostpath,nframe,filename,isGhost);
			}
			else
			{
				csvline=csvline+",N";
				writeGhost(contours,ghostmask,img,noghostpath,nframe,filename,isGhost);
			}
#endif
		}
		csvline=csvline+"\n";
		return csvline;
	}

	void outputFinalResult(vector<Point> contour, Mat & ghostmask, Mat img, int nframe, Rect rec=Rect(), bool isGhost=false, string titletext=string(), string sectiontext=string())
	{
		vector<vector<Point> > contours;
		contours.push_back(contour);
		string recstr=rectToString(rec);
		string framestr = u.intToString(nframe);
		string ghostpath=outpath+"/ghost/";
#ifdef REPORT_RESULT
		char charname[20];
		sprintf(charname, "%06d", nframe);
		string filename="bin"+string(charname);
		string noghostpath=outpath+"/ghost/";
#else
		string filename=framestr+"_"+recstr;
		string noghostpath=outpath+"/noghost/";
#endif

#ifdef OUTPUTALL
		if(isGhost)
			writeGhost(contours,ghostmask,img,ghostpath,nframe,filename,isGhost,titletext,sectiontext);
		else
			writeGhost(contours,ghostmask,img,noghostpath,nframe,filename,isGhost,titletext,sectiontext);
#endif
	}

	void loadSvmModel(string xmlfile)
	{
		string model=string("svm");
		if(u.fileExists(xmlfile))
			svm.load(xmlfile.c_str(),model.c_str());
	}

	void loadBayesModel(string yamlfile)
	{
		string model=string("bayes");
		if(u.fileExists(yamlfile))
			bayes.load(yamlfile.c_str(),model.c_str());
	}

	void loadRateSvmModel(string xmlfile)
	{
		string model=string("svm");
		if(u.fileExists(xmlfile))
			ratesvm.load(xmlfile.c_str(),model.c_str());
	}

	void setWorld(const Rect& world) {
		this->world = world;
		diagonal = sqrt(
				world.height * world.height + world.width * world.height);
		minDiagDistance = 0.02 * diagonal;

	}

	void setWorld(Mat & src) {
		this->world = Rect(0,0,src.cols,src.rows);
		diagonal = sqrt(
				world.height * world.height + world.width * world.height);
		minDiagDistance = 0.02 * diagonal;

	}

	int getMinArea() const {
		return minArea;
	}

	void setMinArea(int minarea) {
		minArea = minarea;
	}

	int getPatchArea() const {
		return patchArea;
	}

	void setPatchArea(int patchArea) {
		this->patchArea = patchArea;
	}


	void setExtinput(const string& extinput) {
		this->extinput = extinput;
	}

	void setInputpath(const string& inputpath) {
		this->inputpath = inputpath;
	}

	void setInputPrefix(const string& inputPrefix) {
		this->inputPrefix = inputPrefix;
	}

	void setExtmask(const string& extmask) {
		this->extmask = extmask;
	}

	void setMaskpath(const string& maskpath) {
		this->maskpath = maskpath;
	}

	void setMaskprefix(const string& maskprefix) {
		this->maskprefix = maskprefix;
	}

	void setOutpath(const string& outpath) {
		this->outpath = outpath;
	}

	string getSegmentationName() {
		return segmentationName;
	}

	void setSegmentationName(string& segmentationName) {
		this->segmentationName = segmentationName;
	}

	string getTestCaseName() {
		return testCaseName;
	}

	void setTestCaseName(string& testCaseName) {
		this->testCaseName = testCaseName;
	}

	int getTotalGhosts() const {
		return totalGhosts;
	}

	int getTotalNonBlack() const {
		return totalNonBlackFrames;
	}

	int getTotalAnalyzed() const {
		return totalBlobsAnalyzed;
	}

	void setFrameNo(int frameNo) {
		this->frameNo = frameNo;
	}

	const string& getSegmentationName() const {
		return segmentationName;
	}

	void setSegmentationName(const string& segmentationName) {
		this->segmentationName = segmentationName;
	}

	const string& getTestCaseName() const {
		return testCaseName;
	}

	void setTestCaseName(const string& testCaseName) {
		this->testCaseName = testCaseName;
	}

	string rectToString(Rect & r)
	{
		string rs;
		stringstream sstm; //create a stringstream
		sstm.str("");
		sstm.clear();
		sstm << r.x<< "_" << r.y << "_"  << r.width << "_" << r.height;
		rs = sstm.str();
		return rs;
	}


	vector<KeyPoint> getRandomKeyPoints(int w, int h, int grid)
	{
		int th=5;
		vector<KeyPoint> vkp;
		vector<Point2f> pointset;
		int i;
		gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);
		int n=min(10,w*h/(grid*grid));
		for (i = 0; i < n; i++)
		{
			double v[2];
			gsl_qrng_get (q, v);
			Point2f p(v[0]*w, v[1]*h);
        	pointset.push_back(p);
		}

		u.removedupes(pointset);
		for(uint j=0; j<pointset.size(); j++)
		{
			KeyPoint kp(pointset[j],th);
			vkp.push_back(kp);
		}

		gsl_qrng_free (q);
		return vkp;
	}

	int getTotalNoGhosts() const {
		return totalNoGhosts;
	}

private:

	bool isRecorded(int nframe,string rectstr, string segmentation, string testcase)
	{
		return false;
	}

	int countVotes(vector<float> & v)
	{
		int countVote=0;
		for(int j=0; j<5; j++)
			countVote = v[j]>0.95 ? countVote+1 : countVote;
		for(int j=5; j<45; j++)
			countVote = v[j]>0.85 ? countVote+1 : countVote;
		for(int j=45; j<60; j++)
			countVote = v[j]>0.3 ? countVote+1 : countVote;
		for(int j=60; j<65; j++)
			countVote = v[j]>0.6 ? countVote+1 : countVote;
		return countVote;
	}

	vector<Point> getRandomPoints(int w, int h, int grid)
	{
		vector<Point> vp;
		int i;
		gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);
		int n=min(30,w*h/(grid*grid));
		for (i = 0; i < n; i++)
		{
			double v[2];
			gsl_qrng_get (q, v);
			cv::Point p(v[0]*w, v[1]*h);
			vp.push_back(p);
		}
		gsl_qrng_free (q);
		return vp;
	}

	void randomPatches(int height, int width, int n, vector<Rect>& patch_vec) {
		vector<Point> sp = getRandomPoints(width,height,n);
		Rect tworld(0,0,width,height);
		int slength=MIN(height,width)/(n*2);
		for(uint i=0; i<sp.size(); i++)
		{
			Rect r(sp[i].x-slength,sp[i].y-slength,slength,slength);
			patch_vec.push_back(r);
		}
	}

	Mat buildMask(Mat src, int kernelSize)
	{
		Mat dest;
		u.capHoles(src);
		Mat E, D;
		erode(src,E,getStructuringElement(MORPH_ELLIPSE, Size(kernelSize,kernelSize)),cv::Point(-1,-1),1);
		dilate(src,D,getStructuringElement(MORPH_ELLIPSE, Size(kernelSize,kernelSize)),cv::Point(-1,-1),1);
		int countnz = countNonZero(E);
		if(countnz<30)
			return dest;
		dest=Mat(src.size(),CV_8U);
		Mat eroCrown, dilCrown;
		absdiff(E,src,eroCrown);
		absdiff(D,src,dilCrown);
		dest.setTo(GC_PR_BGD);
		dest.setTo(GC_FGD,E);
		//dest.setTo(GC_PR_FGD,eroCrown);
		//dest.setTo(GC_PR_BGD,dilCrown);
		return dest;
	}

	int findObjects(Mat& sumth, Mat src)
	{
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
		findContours(expEdges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
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

	Rect expandRect(Rect &r, float rate)
	{
		int nwidth, nheight, nx, ny;
		if(r.width*rate<4)
			nwidth=r.width+4;
		else
			nwidth=floor(r.width*(1+rate));

		if(r.height*rate<4)
			nheight=r.height+4;
		else
			nheight=floor(r.height*(1+rate));

		nx=r.x-ceil((double)abs(nwidth-r.width)/2);
		ny=r.y-ceil((double)abs(nheight-r.height)/2);
		if(nx<0)
			nx=0;
		if(ny<0)
			ny=0;
		Rect nrect(nx,ny,nwidth,nheight);
		return nrect;
	}

	std::vector<float> apply_permutation( std::vector<float> & vec, std::vector<int> & p)
					{
		std::vector<float> sorted_vec(p.size());
		std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
		return sorted_vec;
					}

	std::vector<int> sort_permutation(vector<float> const & w)
					{
		std::vector<int> p(w.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),[&](int i, int j){ return w[i]<w[j]; });
		return p;
					}

	std::vector<int> apply_permutation( std::vector<int> & vec, std::vector<int> & p)
					{
		std::vector<int> sorted_vec(p.size());
		std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
		return sorted_vec;
					}

	std::vector<int> sort_permutation(vector<int> const & w)
					{
		std::vector<int> p(w.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),[&](int i, int j){ return w[i]<w[j]; });
		return p;
					}
	float calcDistance(Rect r1, Rect r2)
	{
		int cx1 = r1.x+r1.width/2;
		int cy1 = r1.y+r1.height/2;
		int cx2 = r2.x+r2.width/2;
		int cy2 = r2.y+r2.height/2;
		int difx=cx1-cx2;
		int dify=cy1-cy2;
		float dist = sqrt(difx*difx+dify*dify);
		return dist;
	}

	vector<Rect> regions;
	UtilCpp u;
	int appearenceModel;
	Rect world;
	Mat lastFrame[4];
	Mat frameDiff;
	//int imgdiag;
	int minArea;
	int patchArea;
	Mat ghost;
	//int votesTh;
	Mat openEdges;
	Mat edgesRGB;
	int diagonal;
	int minDiagDistance;
	string inputpath;
	string inputPrefix;
	string extinput;
	string maskpath;
	string maskprefix;
	string extmask;
	string segmentationName;
	string testCaseName;
	string outpath;
	int totalGhosts;
	int totalNoGhosts;
	int totalNonBlackFrames;
	int totalBlobsAnalyzed;
	int frameNo;
	int difFrame;
	//mysqldb mydb;
    CvSVM svm;
    CvSVM ratesvm;
    CvNormalBayesClassifier bayes;
    bool createdHeader;
    float initResizeFactor, incResizeFactor;
    int initCurrpatcharea, incCurrpatcharea;
    float initWeight, incWeight;
    float initUpth, incUpth;
    float initDifth, incDifth;
    float initDisrate, incDisrate;
    vector<bool> maxDefault;
    bool createdDescriptor;
    float f3Aver;
    float f4Aver;
};
};
#endif /* HESSTRACKER_H_ */
