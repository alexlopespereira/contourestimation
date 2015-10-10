/*
 * HessTracker.h
 *
 *  Created on: Feb 9, 2014
 *      Author: alex
 */

#ifndef HESSTRACKER_H_
#define HESSTRACKER_H_
#include <vector>
#include <deque>
#include <algorithm>
#include <string>
#include "opencv2/opencv.hpp"
#include "UtilCpp.h"
#include "HessParticle.h"
#include "cmath"
#include "GhostDetector.h"

using namespace Util;
using namespace std;
using namespace cv;
static int autoIncObjID;


static int particle_cmpHT( const void* p1, const void* p2 );

//int r1,r2,r3,r4,r5,r6,r7;

namespace HessTracking
{
class HessTracker {
public:
	HessTracker(){
		totalNonBlackFrames=0;
		totalGhosts=0;
		totalNoGhosts;
		totalBlobsAnalyzed=0;
		countInvalid=0;
		createdHeader=false;
		minArea=200;
		frameNo=0;
		nTrajectories=0;
		imgdiag=0;
		validWeight=0;
		ih=iw=0;
		minGhostArea=200;
		p_perObject=5;
		patchArea=200;
		gsl_rng_env_setup();
		rng = gsl_rng_alloc( gsl_rng_mt19937 );
		gsl_rng_set( rng, time(NULL) );
	}

	~HessTracker(){
		vector<trajectory *>::iterator it;
		int c=0;
		for(it = trajectories.begin(); it!=trajectories.end(); it++)
		{
			if(*it!=0)
				delete (*it);
			c++;
		}
	}

	void init(Mat & frame, Mat & mask, Mat & bgFrame, int particlesPerObject)
	{
		//r1=r2=r3=r4=r5=r6=r7=0;
		deadtracklife=0;
		gd.setOutpath(outpath);
		gd.setExtinput(extinput);
		gd.setInputpath(inputpath);
		gd.setInputPrefix(inputPrefix);
		gd.setMaskpath(maskpath);
		gd.setExtmask(extmask);
		gd.setMaskprefix(maskprefix);
		//gd.setSegmentationName(findSegmentationName(maskpath));
		//gd.setTestCaseName(findLastToken(maskpath));

		vpaths = vector<string>{maskpath+maskprefix,extmask,inputpath+inputPrefix,extinput,outpath};
		//votesTh=20;
		//appearenceModel=HISTOGRAM;
		validWeight=3;
		//setMinArea(frame);
		vector<int> vAreas;
		vector<int> vLen;
		vector<Point2f> cofm;
		vector<vector<Point> > meaningfulContours;
		vector<vector<Point> > contoursHull = gd.getContours(mask, regions, cofm, meaningfulContours, vAreas, vLen);

		p_perObject=particlesPerObject;
		//nTrajectories=regions.size();
		//nbins=16;
		iw=frame.cols;
		ih=frame.rows;
		autoIncObjID=0;
		world = Rect(0,0,frame.cols,frame.rows);
		Mat filteredRGB=frame;
//		bilateralFilter(frame, filteredRGB, 16, 32, 10);
		Mat hsvf = u.bgr2hsv( filteredRGB );
		for (uint i=0; i<regions.size(); i++)
		{
			trajectories.push_back(new trajectory);
			trajectories[i]->object.setMinArea(minArea, patchArea);
			trajectories[i]->object.objectID=autoIncObjID++;
			trajectories[i]->object.initParticles(hsvf, mask, regions[i], contoursHull[i], meaningfulContours[i], cofm[i], particlesPerObject,INITIAL_CREATION);
			trajectories[i]->object.setStartFrame(frameNo);
			trajectories[i]->object.setWorld(world);
			trajectories[i]->object.lifeTrack.push_back(frameNo);
			trajectories[i]->object.contourHistory.push_back(meaningfulContours[i]);
			trajectories[i]->object.setArea(vAreas[i]);
			trajectories[i]->object.setLength(vLen[i]);
			//trajectories[i]->object.countVotes=0;
//			Mat imgobj = Mat(hsvf,regions[i]);
//			Mat maskobj = Mat(mask,regions[i]);
//			trajectories[i]->object.objImg = imgobj.clone();
//			trajectories[i]->object.objMask = maskobj.clone();
		}
#ifdef REPORT_RESULT
			gd.outputFinalResult(vector<Point>(),mask,frame,frameNo,Rect());
#endif
			nTrajectories=trajectories.size();
		if(nTrajectories>0)
		{
			removeGhostTrajectories(filteredRGB, hsvf, mask, bgFrame);
		}
		frameNo++;
	}

	void next(Mat & frameRGB, Mat & mask, Mat & resultMask, Mat & bgFrame)
	{
		uint j;
		Mat hsvf = u.bgr2hsv( frameRGB );
		vector<int> toRemove;
		vector<int> vAreas;
		vector<int> vLen;
		vector<Point2f> cofm;
		vector<vector<Point> > meaningfulContours;
		vector<vector<Point> > contoursHull = gd.getContours(mask, regions, cofm, meaningfulContours, vAreas, vLen);
#ifdef REPORT_RESULT
//		//gd.outputFinalResult(vector<Point>(),mask,frameRGB,frameNo,Rect());
		string noghostpath=outpath+"/ghost/bin";
		u.writeImg(noghostpath,mask,frameNo,"png");
#endif

		/* perform prediction and measurement for each particle */
		for( j = 0; j < trajectories.size(); j++ )
		{
			trajectories[j]->object.transition(iw, ih, rng);
			//trajectories[j]->object.adjustParticles(hsvf, mask);
			trajectories[j]->object.updateWeights(hsvf, mask);
			if(!trajectories[j]->object.isConsistent(mask) && trajectories[j]->object.weight<-15)
			{
				toRemove.push_back(trajectories[j]->object.objectID);
			}
			trajectories[j]->object.normalize_weights();
			trajectories[j]->object.resample();
		}

		for(uint i=0; i<toRemove.size(); i++)
		{
			removeTrack(toRemove[i],vpaths);
		}
		nTrajectories=trajectories.size();

		addHeterogObjects(hsvf, mask, regions, contoursHull, meaningfulContours, cofm, vAreas, vLen, regions.size());

//		if(regions.size()==0)
//		{
//			string noghostpath=outpath+"/ghost/bin";
//			u.writeImg(noghostpath,mask,frameNo,"png");
//		}

		if(nTrajectories>0)
		{
			removeGhostTrajectories(frameRGB, hsvf, mask, bgFrame);
		}


		frameNo++;
		return;
	}

	void showResults(Mat & frame)
	{
		for (int i=0; i<nTrajectories; i++)
		{
			Scalar color;
			color = CV_RGB(255,0,0);
			if (trajectories[i]->object.weight >= 0)
			{
				color=Scalar(255,0,0);
			}
			else
				color=Scalar(10,240,10);

			if (trajectories[i]->object.match)
			{
				int p=SHOW_SELECTED;
				trajectories[i]->object.displayParticles( frame, CV_RGB(255,0,0), color , p);
				//rectangle(frame,trajectories[i]->object.observedRect,Scalar(180,055,50));
				vector<vector<Point> > contourVec;
				contourVec.push_back(trajectories[i]->object.getContourReal());
				drawContours(frame, contourVec, 0, Scalar(10,055,150));
				CvFont font;
				cvInitFont(&font,CV_FONT_HERSHEY_PLAIN|CV_FONT_ITALIC,1,1,0,1);
				char buffer[4], reason[4];
				sprintf (buffer, "%d",trajectories[i]->object.objectID );
				sprintf (reason, "%d",trajectories[i]->object.reason );
				Point orig = Point( cvRound(trajectories[i]->object.tparticles[0].x)+5, cvRound(trajectories[i]->object.tparticles[0].y)+5 );
				Point origReason = Point( cvRound(trajectories[i]->object.tparticles[0].x)+15, cvRound(trajectories[i]->object.tparticles[0].y)+15 );
				std::string text(buffer);
				std::string textReason(reason);
				int fontFace = cv::FONT_HERSHEY_PLAIN;
				double fontScale = 1;
				int thickness = 1;
				int baseline=0;
				baseline += thickness;
				cv::putText(frame, text, orig, fontFace, fontScale, Scalar(255,0,0), thickness, 8);
				cv::putText(frame, textReason, origReason, fontFace, fontScale, Scalar(0,255,0), thickness, 8);
			}
		}
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

	void addHeterogObjects(Mat & frameHSV, Mat & mask, vector<Rect> & regions, vector<vector<Point> > contoursHull, vector<vector<Point> > contoursReal, vector<Point2f> & vcmass, vector<int> & vAreas, vector<int> & vLen, int nRegions)
	{
		HessParticle particl;
		particl.setWorld(world);
		particl.setMinArea(minArea, patchArea);
		Rect inter[3];//, inter2, inter3;
		vector<uchar> trFound(nTrajectories);
		vector<uchar> toMerge(nTrajectories);
		vector<int> interVector(nTrajectories);
		vector<int> orderTraj(nTrajectories);
		//vector<uchar> usedReg(regions.size());
		vector<RotatedRect> minEllipse( contoursHull.size() );

		//std::fill(usedTraj.begin(), usedTraj.end(),0);
		std::fill(trFound.begin(), trFound.end(),0);
		std::fill(toMerge.begin(), toMerge.end(),0);
		//std::fill(usedReg.begin(), usedReg.end(),0);
		vector<int> toRemove;

		int rs=0, tr=-1, i;
		float d=1000000, dm=0;
		int tmpNTrajectories=nTrajectories;
		for (i=0; i<tmpNTrajectories; i++)
		{
			trFound[i]=0;
			trajectories[i]->object.match=false;
		}
		if (nRegions<1)
		{
			for (uint j=0; j<trFound.size(); j++)
			{
				trajectories[j]->object.match=false;
				trajectories[j]->object.weight+=-2;
				if (trajectories[j]->object.weight<-14 && trajectories[j]->object.isOld(frameNo,15))
					toRemove.push_back(trajectories[j]->object.objectID);
			}
			for(uint i=0; i<toRemove.size(); i++)
			{
				removeTrack(toRemove[i],vpaths);
			}
			nTrajectories=trajectories.size();
			return;
		}

		if(tmpNTrajectories > 0)
		{
			for (i=0; i<nRegions; i++)
			{
				dm=0;
				interVector.clear();
				orderTraj.clear();
				for (int j=0; j<tmpNTrajectories; j++)
				{
					inter[0] = trajectories[j]->object.tparticles[0].r & regions[i];
					inter[1] = trajectories[j]->object.tparticles[1].r & regions[i];
					inter[2] = trajectories[j]->object.tparticles[2].r & regions[i];
					int tmparea[3] = {inter[0].width*inter[0].height, inter[1].width*inter[1].height,inter[2].width*inter[2].height};
					int tmpindex = tmparea[0] > tmparea[1] ? 0 : 1;
					tmpindex = tmparea[tmpindex] > tmparea[2] ? tmpindex : 2;
					//float dist = calcDistance(trajectories[j]->object.tparticles[0].r,regions[i]);
					if(inter[tmpindex].width==0 || inter[tmpindex].height==0 || trFound[j]==1 /*|| dist>(regions[i].width+regions[i].height)*/)
						continue;

					d=inter[tmpindex].width * inter[tmpindex].height;
					interVector.push_back(d);
					orderTraj.push_back(j);
				}

				if(interVector.size()==0)
				{
					nTrajectories++;
#ifdef MOMENTS
					addNewObject(frameHSV, regions[i], contoursHull[i], contoursReal[i], vcmass[i], vAreas[i], vLen[i], mask, REASON_2);
#else
					addNewObject(frameHSV, regions[i], contoursHull[i], Point2f(), mask, REASON_2);
#endif
					continue;
				}
				sortTwoVectors(interVector,orderTraj);
				while(orderTraj.size()>0)
				{
					tr=orderTraj.back();
					dm=interVector.back();
					orderTraj.pop_back();
					interVector.pop_back();
					bool isEllipse=false;
					RotatedRect pEllipse;
					bool sameType=false;
					if(trajectories[tr]->object.contourHull.size()<5)
					{
						Point2f cent;
						float radius;
						sameType=contoursHull[i].size()<5;
						minEnclosingCircle(Mat(contoursHull[i]), cent, radius);
						minEllipse[i]=RotatedRect(cent,Size2f(radius,radius),0);
						minEnclosingCircle( Mat(trajectories[tr]->object.contourHull), cent, radius );
						pEllipse=RotatedRect(cent,Size2f(radius,radius),0);
						isEllipse=false;
					}
					else
					{
						sameType=contoursHull[i].size()>=5;
						if(sameType)
							minEllipse[i] = fitEllipse( Mat(contoursHull[i]) );
						else
							minEllipse[i] = RotatedRect(Point2f(-1,-1),Size2f(-1,-1),-1);
						pEllipse = fitEllipse( Mat(trajectories[tr]->object.contourHull) );
						isEllipse=true;
					}
					float similarity = particl.likelihood(frameHSV, mask,regions[i],
							trajectories[tr]->object.tparticles[0].vhistos, trajectories[tr]->object.getNrowsCols(), trajectories[tr]->object.tparticles[0].n);
					//cout << "similarity: " << similarity << endl;
					float displ = sqrt(pow(trajectories[tr]->object.cOfMass.x - vcmass[i].x,2)+pow(trajectories[tr]->object.cOfMass.y - vcmass[i].y,2));
					bool atBorder = trajectories[tr]->object.rectAtBorder(regions[i]);
					bool ruleAtMiddle = sameType && !atBorder && (compareEllipses(minEllipse[i], pEllipse,displ,0.3,isEllipse));
					bool ruleAtBorder = sameType && atBorder && (compareEllipses(minEllipse[i], pEllipse,displ,0.55,isEllipse));

	//				Mat maskc = mask.clone();
	//				ellipse(maskc,minEllipse[i],Scalar(120,0,0));
	//				ellipse(maskc,pEllipse,Scalar(200,0,0));
	//				#ifdef DEBUG
	//					imshow("ellipse",maskc);
	//				#endif
	//				u.writeRGB("ellipse/ellipse",maskc,frameNo*1000+trajectories[tr]->object.objectID);
					long a1 = trajectories[tr]->object.tparticles[0].width*trajectories[tr]->object.tparticles[0].height;
					long a2 = regions[i].width*regions[i].height;
					bool ruleMin = (dm>0.30*a1 && dm>0.30*a2) && (similarity > 0.30) &&
							(ruleAtBorder || ruleAtMiddle );

					int countVote=0;
					countVote=(dm>0.7*a1 && dm>0.7*a2) ? countVote+1 : countVote;
					countVote=(similarity > 0.85) ? countVote+1 : countVote;
					countVote=(compareEllipses(minEllipse[i], pEllipse,displ,0.2,isEllipse) && sameType) ? countVote+1 : countVote;
					bool ruleVote=countVote>=2;
					if (ruleMin || ruleVote) //TODO: melhorar essa regra.
					{
						if(trajectories[tr]->object.changeLastParticle(frameHSV, mask, regions[i]))
						{
							trajectories[tr]->object.match=true;
							trajectories[tr]->object.setContourReal(contoursReal[i]);
							trajectories[tr]->object.contourHistory.push_back(contoursReal[i]);
							trajectories[tr]->object.lifeTrack.push_back(frameNo);
							trajectories[tr]->object.cmtrackx.push_back(vcmass[i].x);
							trajectories[tr]->object.cmtracky.push_back(vcmass[i].y);
							trajectories[tr]->object.matchCount++;
							trajectories[tr]->object.setObservedRect(regions[i]);
							trajectories[tr]->object.setContourHull(contoursHull[i]);
							trajectories[tr]->object.weight+=1;
							trajectories[tr]->object.setArea(vAreas[i]);
							if(trajectories[tr]->object.weight>validWeight)
								trajectories[tr]->object.weight=validWeight;
	#ifdef MOMENTS
							trajectories[tr]->object.cOfMass=vcmass[i];
	#endif
							trFound[tr]=1;
							rs++;
							break;
						}
						else
						{
							trFound[tr]=0;
							trajectories[tr]->object.match=false;
						}
					}
				}
				if(trFound[tr]!=1)
				{
					trFound[tr]=0;
						nTrajectories++;
#ifdef MOMENTS
						addNewObject(frameHSV, regions[i], contoursHull[i], contoursReal[i], vcmass[i], vAreas[i], vLen[i], mask, REASON_3);
#else
						addNewObject(frameHSV, regions[i], contoursHull[i], Point2f(), vAreas[i], mask, REASON_3);
#endif
				}
			}
		}
		else
		{
			for (i=0; i<nRegions; i++)
			{
				nTrajectories++;
#ifdef MOMENTS
				addNewObject(frameHSV, regions[i], contoursHull[i], contoursReal[i], vcmass[i], vAreas[i], vLen[i], mask, REASON_4);
#else
				addNewObject(frameHSV, regions[i], contoursHull[i], Point2f(), vAreas[i], mask, REASON_4);
#endif
			}
		}

		for (uint j=0; j<trFound.size(); j++)
		{
			if (trFound[j]==1 )
			{
				if (trajectories[j]->object.weight<1)
					trajectories[j]->object.weight+=1;
			}
			else
			{
				trajectories[j]->object.match=false;
				trajectories[j]->object.weight+=-2;
			}
		}

		int countRemoved=0;

		for (uint k=0; k<nTrajectories; k++)
		{
			//mergeTrack();
			if (trajectories[k]->object.weight<-14 && trajectories[k]->object.isOld(frameNo,15))
			{
				toRemove.push_back(trajectories[k]->object.objectID);
				countRemoved++;
			}

		}
		for(uint i=0; i<toRemove.size(); i++)
		{
			removeTrack(toRemove[i],vpaths);
		}

		nTrajectories=trajectories.size();
	}

	bool compareEllipses(RotatedRect & e1, RotatedRect & e2, float displ, float th, bool isEllipse)
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
		bool ruleMin;

		if(isEllipse)
			ruleMin = isntDisplacedV && simSizeV && simAngleV;
		else
			ruleMin = isntDisplacedV && simSizeV;

		bool isntDisplacedM = (displ< max(e2.size.width*thDecd,e2.size.height*thDecd)) && (displ<max(e1.size.width*thDecd,e1.size.height*thDecd));
		bool simSizeM = sizeRatio.width<thDecd && sizeRatio.height<thDecd;
		bool simAngleM = difAngle<60*thDecd;
		int countVote=0;
		countVote = isntDisplacedM == true ? countVote+1 : countVote;
		countVote = simSizeM == true ? countVote+1 : countVote;
		bool ruleVote;
		if(isEllipse)
		{
			countVote = simAngleM == true ? countVote+1 : countVote;
			ruleVote = countVote>=2;
		}
		else
			ruleVote = countVote>=1;

		return (ruleMin || ruleVote);
	}

	void printRepeated(vector<float> numbers, int n)
	{
		for(int i=0; i<n; i++)
		{
			cout << numbers[0]  << endl;
		}
	}

	bool evalTrackHistory(HessParticle * p, int * reason, string & featuresline, string & displine,vector<string> paths)
	{
//		float medianStdDevs = u.median(p->vstdevs);
//		bool displaced =p->getDisplaced();
		//vector<float> medianf5;
		vector<Vec3f> medianFeatures;
		medianFeatures = getMedian(p->ftvectotal,true);
		//medianf5 = getMedian(p->f5features);
//		vector<float> numbers;
//		numbers.push_back(medianFeatures[0][0]);
//		printRepeated( numbers, p->lifeTrack.size());
		bool isGhost;
		//int reason=0;

//		displine=u.intToString(p->objectID)+":"
//								+"c"+u.intToString(p->analyzedCount)+","
//								+"d"+u.floatToString(p->acumDisplacement)+","
//								+"a"+u.floatToString(p->acumAreaChanges)+","
//								+"p"+u.floatToString(p->psr16);//+","
								//+"s"+u.floatToString(medianStdDevs)+"\n";

#ifdef EVAL_NAIVE_BAYES
		vector<float> v(2);
		v[0]=medianFeatures[0].val[0];
		v[1]=medianFeatures[0].val[1];
		isGhost = gd.isGhostBayes(v);
#else
		uint thlife=0;
			float fc = medianFeatures[0].val[0];
			if(fc<0 && p->lifeTrack.size()>thlife)
				isGhost=true;
			else
				isGhost = false;//gd.svmGhostPrediction(medianFeatures);
#endif
			featuresline=u.floatToString(medianFeatures[0].val[0]);
//		}
//#endif
		return isGhost;
	}

	void printCSVline()
	{
		u.printVector(csvlinevec);
	}

	bool removeTrack(int ID, vector<string> paths)
	{
		int foundID=-1;
		for (uint k=0; k<trajectories.size(); k++)
		{
			if(trajectories[k]->object.objectID==ID)
			{
				foundID=k;
				break;
			}
		}
		//cout << endl;
		if(foundID!=-1)
		{
			deadtracklife+=trajectories[foundID]->object.lifeTrack.size();
			int reason=0;
			string featuresline;
			string displine;
			bool isGhost=false;

			isGhost=evalTrackHistory(&trajectories[foundID]->object,&reason,featuresline,displine,paths);
			string g = isGhost ? string("G") : string("N");
			string recstr=gd.rectToString(trajectories[foundID]->object.observedRect);

			cout << trajectories[foundID]->object.lifeTrack.front() << "-" << trajectories[foundID]->object.lifeTrack.back() << "," <<  recstr << "," << featuresline << ":" << g << endl;
//			u.drawText(trajectories[foundID]->object.roiObj,TITLE,g);
//			imshow("roiObj",trajectories[foundID]->object.roiObj);
//			if(frameNo>1315)
//				waitKey(0);
#ifdef TRAINING
			//float ptheta= trajectories[foundID]->object.countVotes/(float)(trajectories[foundID]->object.matchCount+1);
			string csvline;
			HessParticle jp = trajectories[foundID]->object;
			int lastMatchFrame = jp.lifeTrack.back();
			Mat foo;
			csvline=outputResultBlobToVector(&jp,foo,foo,lastMatchFrame,isGhost);
			csvlinevec.push_back(csvline);
			u.appendToFile(csvlinevec,outpath+"/features.csv",false,1);
			vector<Vec3f> medianFeatures = getMedian(jp.ftvectotal,true,false);
			string g = isGhost ? string("G") : string("N");
			cout << jp.objectID << "," << jp.lifeTrack.size() << "," << medianFeatures[0].val[0] << "," << g << endl;
			csvlinevec.clear();
#else
			if(isGhost)
			{
				totalGhosts+=trajectories[foundID]->object.lifeTrack.size();
#ifdef REPORT_RESULT
			clearGhosts(&trajectories[foundID]->object,paths,isGhost,true,string(""),string(""));
#else
			clearGhosts(&trajectories[foundID]->object,paths,isGhost,false,displine,featuresline);
#endif
#ifdef DEBUG
//				int nf = trajectories[foundID]->object.lifeTrack.back();
//				Mat img = u.getFrame(false, paths[2], nf, 6, paths[3]);
//				//Mat img=frameRGB.clone();
//				//featuresline=u.floatToString(medianFeatures);
//				//featuresline=u.intToString(trajectories[foundID]->object.objectID)+":"+featuresline+","+ u.floatToString(medianStdDevs);
//				featuresline=u.intToString(reason)+"="+featuresline;
//				u.drawText(img,SECTION,featuresline);
//				vector<vector<Point> > contourVec;
//				contourVec.push_back(trajectories[foundID]->object.getContourReal());
//				drawContours( img, contourVec, 0, Scalar(255,0,0), 1);
////				imshow( "Ghost", roiD );
//				imshow( "GhostImg", img );
//				waitKey(0);
#endif
			}
			else
			{
				trajectories[foundID]->object.notGhost++;
				totalNoGhosts+=trajectories[foundID]->object.lifeTrack.size();
			}
#endif

			vector<trajectory *>::iterator it = trajectories.begin()+foundID;
			trajectory * tr = *it;
			trajectories.erase(it);
			delete (tr);
			nTrajectories=trajectories.size();
		}
		return foundID!=-1;
	}

	void writeRemainingLines()
	{
		u.appendToFile(csvlinevec,outpath+"/features.csv",false,1);
	}

	void emptyTrash()
	{
		vector<trajectory *>::iterator it;
		for(it = trash.begin(); it!=trash.end(); it++)
		{
			if(*it!=0)
				delete (*it);
		}
		trash.clear();
	}

	vector<trajectory *> getTrash(){
		return trash;
	}

//	void mergeTrack()
//	{
//		float d=0;
//		vector<int> toRemove;
//
//		for (int i=0;i<nTrajectories-1;i++)
//		{
//			for (int j=i+1;j<nTrajectories;j++)
//			{
//				Rect inter = trajectories[j]->object.tparticles[0].r & trajectories[i]->object.tparticles[0].r;
//				if(inter.width==0 && inter.height==0)
//					continue;
//				d=inter.width * inter.height;
//
//				long a1 = trajectories[i]->object.tparticles[0].width*trajectories[i]->object.tparticles[0].height;
//				long a2 = trajectories[j]->object.tparticles[0].width*trajectories[j]->object.tparticles[0].height;
//				if (d>0.7*a1 || d>0.7*a2)
//				{
//					if (trajectories[i]->object.startFrame<=trajectories[j]->object.startFrame)
//					{
//						trajectories[i]->object.weight=1;
//						trajectories[j]->object.weight-=1;
//						//toRemove.push_back(j);
//					}
//					else
//					{
//						trajectories[j]->object.weight=1;
//						trajectories[i]->object.weight-=1;
//						//toRemove.push_back(i);
//					}
//				}
//			}
//		}
////		for(int i=0; i<toRemove.size(); i++)
////		{
////			removeTrack(toRemove[i]);
////		}
//	}

	void addNewObject(Mat & frameHSV, Rect & r, vector<Point> contHull, vector<Point> contReal, Point2f cmass, int currarea, int len, Mat & mask, int reason)
	{
		trajectories.push_back(new trajectory);
		trajectories.back()->object.startFrame=frameNo;
		trajectories.back()->object.setMinArea(minArea,patchArea);
		trajectories.back()->object.objectID=autoIncObjID++;
		trajectories.back()->object.initParticles(frameHSV, mask, r, contHull, contReal, cmass, p_perObject, reason);
		trajectories.back()->object.weight=0.5;
		trajectories.back()->object.setWorld(world);
		trajectories.back()->object.match=true;
		trajectories.back()->object.lifeTrack.push_back(frameNo);
		trajectories.back()->object.contourHistory.push_back(contReal);
		trajectories.back()->object.setArea(currarea);
		trajectories.back()->object.setLength(len);
		//trajectories.back()->object.countVotes=0;
	}


	vector<Vec3f> getMedian(vector<vector<float> > vv, bool changedMedian=false, bool lowDefault=true)
	{
		if(vv.size()==0)
			return vector<Vec3f>(0);
		int ncols=vv[0].size();
		int nlines=vv.size();
		vector<Vec3f> result(ncols);
		for(int i=0; i<ncols; i++)
		{
			vector<float> numbers;
			for(int j=0; j<nlines; j++)
			{
					numbers.push_back(vv[j][i]);
			}
			if(numbers.size()>0)
			{
				float med,max,min;
				if(changedMedian && numbers[0]<=1)
					med=u.medianOrLower(numbers);
				else
					med=u.median(numbers);
				//max =  *std::max_element(numbers.begin(),numbers.end());
				min =  *std::min_element(numbers.begin(),numbers.end());
				Vec3f measure(med,/*max,*/min,0);
				result[i]=measure;
//				cout << "size = "<< numbers.size() << endl;
//				u.printVector(numbers);
			}
			else
			{
				result[i]=Vec3f(-1,-1,-1);
//				if(lowDefault)
//					result[i]=0;
//				else
//					result[i]=10000;
			}

		}
		return result;
	}

	float getRate(vector<int> v)
	{
		int count=0;
		int n=v.size();
		for(int i=0; i<n; i++)
		{
			if(v[i]>0)
				count++;
		}
		float rate=count/(float)n;
		return rate;
	}

	string outputResultBlobToVector(HessParticle *p, Mat & ghostmask, Mat img, int nframe, bool isGhost, int reason=0)
	{
		if(p->ftvectotal.size()==0)
			return string();
		//
		string recstr=gd.rectToString(p->observedRect);
		string framestr = u.intToString(nframe);
		string csvline = framestr;
		string filename=framestr+"_"+recstr;
		string ghostpath=outpath+"/ghost/"+filename;
		string noghostpath=outpath+"/noghost/"+filename;
		csvline=csvline+","+recstr;
		csvline=csvline+","+u.intToString(p->objectID);
		//csvline=csvline+","+u.boolToString(p->atBorder());
		csvline=csvline+","+u.intToString(p->analyzedCount);
		float psr1  = MIN(1,p->acumDisplacement / (16 * p->objectDiagonal));
		//float psr8  = MIN(1,p->acumDisplacement / (8 * p->objectDiagonal));
		csvline=csvline+","+u.intToString(p->acumDisplacement);
		csvline=csvline+","+u.floatToString(MIN(1000,p->acumAreaChanges));
		csvline=csvline+","+u.floatToString(psr1);
		//csvline=csvline+","+u.floatToString(psr8);
		//float medianStdDevs = u.median(p->vstdevs);
		//csvline=csvline+","+u.floatToString(medianStdDevs);
		vector<Vec3f> medianFeatures = getMedian(p->ftvectotal,true,false);

		//u.printVector(p->ftvectotal[0]);
		csvline=csvline+","+u.Vec3fToString(medianFeatures[0])+","+u.Vec3fToString(medianFeatures[1])+","+u.Vec3fToString(medianFeatures[2]);
		if(isGhost)
		{
			csvline=csvline+",G";
		}
		else
		{
			csvline=csvline+",N";
		}

		csvline=csvline+"\n";
		return csvline;
	}

	string outputResultBlobToVector(Rect r, vector<float> ftvector, Mat & ghostmask, Mat img, int nframe, bool isGhost, int reason=0)
	{
		string recstr=gd.rectToString(r);
		string framestr = u.intToString(nframe);
		string csvline = framestr;
		string filename=framestr+"_"+recstr;
		string ghostpath=outpath+"/ghost/"+filename;
		string noghostpath=outpath+"/noghost/"+filename;
		csvline=csvline+","+recstr;
//		csvline=csvline+","+u.floatToString(medianStdDevs);
//		vector<float> medianFeatures = getMedian(p->ftvectotal,true,false);
		csvline=csvline+u.floatToString(ftvector);
		if(isGhost)
		{
			csvline=csvline+",G";
		}
		else
		{
			csvline=csvline+",N";
		}

		csvline=csvline+"\n";
		return csvline;
	}

	void clearGhosts(HessParticle * p, vector<string> paths, bool isghost=true, bool reportResult=false, string titletext=string(), string sectiontext=string())
	{
		vector<string> vlines;
		vector<vector<Point> > contours = p->contourHistory;
		vector<int> frameNums = p->lifeTrack;
		std::map<int,bool>::iterator it;
		int count=0;
		for(uint i=0; i<contours.size(); i++)
		{
			count++;
			it=p->cleared.find(frameNums[i]);
			if(it->second)
				continue;
			else
				p->cleared[frameNums[i]]=true;
			Mat ghostmask;
			Mat pastImg;
#ifdef OUTPUTALL
				pastImg = u.getFrame(false, paths[2], frameNums[i], 6, paths[3]);
				if(reportResult)
					ghostmask = u.getFrame(false, outpath+"/ghost/bin", frameNums[i], 6, "png");
				else
					ghostmask = Mat::zeros(pastImg.size(),CV_8U);

			string csvline;
			//string numbers = u.intToString(reason)+":"+text; //u.floatToString(p->acumAreaChanges) + "," + u.floatToString(p->psr8);
			Rect rec = boundingRect( Mat(contours[i]) );
			gd.outputFinalResult(contours[i],ghostmask,pastImg,frameNums[i],
											rec,isghost,titletext,sectiontext);
#endif
			//vlines.push_back(csvline);
		}
		//return vlines;
	}

	void writeMapToFile()
	{
		u.appendToFile(csvlinemap,outpath+"/features.csv");
	}

	void analyseRemainingTrajectories()
	{
		vector<string> csvlinevec;
		Mat foo;
		int index=0;
		bool hasRemoved=true;
		int notChanged=0;
		int lastIndex=0;
		while(nTrajectories>0 && index<nTrajectories)
		{
			hasRemoved = removeTrack(trajectories[index]->object.objectID,vpaths);
			if(!hasRemoved)
			{
				index++;
				notChanged++;
				lastIndex=index;
			}
			else
			{
				index=lastIndex;
			}
		}
		writeRemainingLines();
	}

	void createEdgesStdDev(Mat frameRGB)
	{
		Mat sobedges = frameRGB.clone();
		u.applySobel(sobedges);
		Mat sob8b;
		cvtColor(sobedges, sob8b, CV_RGB2GRAY);
		Scalar mean;
		Scalar std;
		meanStdDev(sob8b,mean,std);
		//bool smallStddev=std[0]<30;
		string stdstr=u.floatToString(std[0])+"\n";
		u.appendToFile(stdstr,outpath+"/edges_stdev.csv");
	}

	float getEdgesStdDev(Mat frameRGB, Mat & mask)
	{
		Mat cont=frameRGB.clone();
		Mat maskCont = u.convertTo3Channels(mask);
		Mat sobedges = frameRGB.clone();
		u.applySobel(sobedges);
		Mat sob8b;
		cvtColor(sobedges, sob8b, CV_RGB2GRAY);
		Scalar mean;
		Scalar std;
		meanStdDev(sob8b,mean,std);
		//string stdstr=u.floatToString(std[0]);
		//u.appendToFile(stdstr,outpath+"/edges_stdev.txt");
		if(countNonZero(mask)<=0 || nTrajectories<=2)
			return std[0];

		vector<float> stdevs;
		for (int i=0;i<nTrajectories;i++)
		{
			if(!trajectories[i]->object.match)
				continue;

			stdevs.push_back(gd.getEdgesStdDev(sob8b,trajectories[i]->object.observedRect));
		}

		if(stdevs.size()==0)
			return std[0];
		else
			return u.median(stdevs);
	}

	int sumLifeTrack()
	{
		int i;
		int s=0;
		for (i=0; i<trajectories.size(); i++)
		{
			s+= trajectories[i]->object.lifeTrack.size();
		}
		return s+deadtracklife;
	}

	void removeGhostTrajectories(Mat frameRGB, Mat frameHSV, Mat & mask, Mat & bgFrame)
	{
		if(countNonZero(mask)<=0)
		{
			return;
		}

//		float stddev = getEdgesStdDev(frameRGB,mask);
//		addToStdDevDeque(stddev);
//		float meanStdDev = getStdDevsMean();

		Mat cont=frameRGB.clone();
		//Mat maskCont = u.convertTo3Channels(mask);
		vector<int> toRemove;
		vector<string> csvlinevec;
		Mat bgHSV= u.bgr2hsv(bgFrame);
		totalNonBlackFrames++;
		vector<Mat> srcMats{frameRGB,frameHSV};
		vector<Mat> srcMatsBg{bgFrame,bgHSV};
		//cout <<"trajectories:" << nTrajectories << "," << trajectories.size() << endl;
		for (int i=0;i<nTrajectories;i++)
		{
			HessParticle p = trajectories[i]->object;
			Mat maskhull = Mat::zeros(mask.size(),CV_8U);
			Mat maskc = Mat::zeros(mask.size(),CV_8U);
			Mat roiD=Mat(maskc,p.observedRect);
			Mat roiS=Mat(mask,p.observedRect);
			roiS.copyTo(roiD);
			vector<vector<Point> > contourVec;
			contourVec.push_back(p.getContourHull());
			drawContours( maskhull, contourVec, 0, Scalar(255), CV_FILLED);
			int cntNonZero=countNonZero(roiS);
			//bool isAtBorder =  p.rectAtBorder(p.observedRect);
			//bool displaced = p.getDisplaced();

			bool earlyRule = cntNonZero<minGhostArea || !p.match;
			if(earlyRule)
			{
				//cout << "earlyrule" << endl;
				continue;
			}

			totalBlobsAnalyzed++;
			trajectories[i]->object.analyzedCount++;
			trajectories[i]->object.roiObj=Mat(frameRGB,p.observedRect).clone();
			vector<float> ftvectotal_in;
			vector<float> ftvectotal_bg;
			vector<string> vhtotal;

			gd.getFeatures(p.observedRect, p.getContourReal(), p.getLength(), maskc, maskhull,srcMats,vflist,ftvectotal_in,vhtotal,i==0);
			gd.getFeatures(p.observedRect, p.getContourReal(), p.getLength(), maskc, maskhull,srcMatsBg,vflist,ftvectotal_bg,vhtotal,i==0);
			vector<float> sub = u.subtract(ftvectotal_bg, ftvectotal_in);
			vector<float> sumsub;
			sumsub.push_back(sub[0]+sub[1]);
//			sub.insert(sub.end(),ftvectotal_in.begin(),ftvectotal_in.end());
			//sub.insert(sub.end(),ftvectotal_bg.begin(),ftvectotal_bg.end());

			//u.printVector(sub,true);
			//ftvectotal = gd.getFeaturesSub(frameRGB,bgFrame,mask,vflist);
			trajectories[i]->object.ftvectotal.push_back(sumsub);

#ifdef CREATE_HEADER
			if(i==0 && !createdHeader)
			{
				createdHeader=true;
				u.appendToFile(vhtotal,outpath+"/header.csv",true);
			}
#endif
		}
		int suml = sumLifeTrack();
		//cout << "totalBlobsAnalyzed:" << totalBlobsAnalyzed << ", " << "sum_lifetr=" << suml <<endl;

		nTrajectories=trajectories.size();
	}

	bool atBorder(Rect r)
	{
		bool atborder = r.x<2 || r.y<2 || (r.x+r.width)>world.width-2 || (r.y+r.height)>world.height-2;
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
			similarity = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, roiF_Rgb, true);
#else
			similarity = p.compareIntExtHistogramsPatches(roiF, roiExpM, dr, er, sqrtArea, roiF_Rgb);
#endif

		return similarity > simTh ? GHOST : VALID_OBJ;
	}
	*/

	void loadBayesModel(string yamlfile)
	{
		gd.loadBayesModel(yamlfile);
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

	void printVector( vector<int> & v)
	{
		printf("\n");
		for(uint i=0; i<v.size(); i++)
		{
			printf("%d ", v[i]);
		}
		fflush(stdout);
	}

	void setOutpath(const string& outpath) {
		this->outpath = outpath;
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

	void setImgdiag(int imgdiag) {
		this->imgdiag = imgdiag;
	}

	void setFrameNo(int frameNo) {
		this->frameNo = frameNo;
	}


	int getMinArea() const {
		return minArea;
	}

	void setMinArea(Mat img, int minarea) {
		imgdiag = sqrt(img.cols*img.cols+img.rows*img.rows);
		minArea = minarea;//std::pow(imgdiag*0.06,2);
		minGhostArea= minarea;
		patchArea=467;
		gd.setMinArea(minArea);
		gd.setPatchArea(patchArea);
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

private:

	void addToStdDevDeque(float f)
	{
		dq_stdevs.push_back(f);
		if(dq_stdevs.size()>3)
			dq_stdevs.pop_front();
	}

	float getStdDevsMean()
	{
		float sum = std::accumulate(dq_stdevs.begin(), dq_stdevs.end(), 0.0);
		float mean = sum / dq_stdevs.size();
		return mean;
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
	std::vector<trajectory *> trajectories;
	std::vector<trajectory *> trash;

	GhostDetector gd;
	vector< vector<int> > vflist;
	//vector< vector<int> > f5list;
	vector<Rect> regions;
	int nTrajectories;
	int frameNo;
	int p_perObject;
	//int nbins;
	gsl_rng* rng;
	UtilCpp u;
	int iw, ih;
	//int appearenceModel;
	Rect world;
	int totalGhosts;
	int totalNoGhosts;
	int totalNonBlackFrames;
	int totalBlobsAnalyzed;
	Mat lastFrame[4];
	int imgdiag;
	int validWeight;
	int countInvalid;
	int minArea;
	int minGhostArea;
	int patchArea;
	//int votesTh;
    bool createdHeader;
    CvSVM svm01;
    CvSVM svmgt1;

	string outpath;
	string inputpath;
	string inputPrefix;
	string extinput;
	string maskpath;
	string maskprefix;
	string extmask;
	string segmentationName;
	string testCaseName;
	map<string,string> csvlinemap;
	vector<string> vpaths;

	deque<float> dq_stdevs;
	vector<string> csvlinevec;
	int deadtracklife;

public:
	int getPatchArea() const {
		return patchArea;
	}

	void setPatchArea(int patchArea) {
		this->patchArea = patchArea;
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

	int getImgdiag() const {
		return imgdiag;
	}

	void setVflist(vector<vector<int> > vflist) {
		this->vflist = vflist;
	}

	int getTotalNoGhosts() const {
		return totalNoGhosts;
	}
};
};
#endif /* HESSTRACKER_H_ */
