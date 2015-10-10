/*
 * Sobel.h
 *
 *  Created on: Dec 2, 2014
 *      Author: alex
 */

#ifndef LBSOBEL_H_
#define LBSOBEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "ULBSP.h"
using namespace Util;
using namespace std;
using namespace cv;

class ULBSobel
{
public:
	ULBSobel(){}
	~ULBSobel(){}
	int xEdgeMag(Mat & src, int th=10, int type=0, int inter=-1){
		//Rect rroi(1,1,src.cols-1,src.rows-1);
		int gx;
		const Mat_<uchar> roi=src;
		ushort f[6];
		vector<KeyPoint> vp;
		if(type==0)
			vp={KeyPoint(2,2,th),KeyPoint(4,2,th),KeyPoint(2,3,th),KeyPoint(4,3,th),KeyPoint(2,4,th),KeyPoint(4,4,th)};
		else
			vp={KeyPoint(2,2,th),KeyPoint(6,2,th),KeyPoint(2,4,th),KeyPoint(6,4,th),KeyPoint(2,6,th),KeyPoint(6,6,th)};
		for (uint i=0; i<vp.size(); i++) {
			const uchar u = src.at<uchar>(vp[i].pt.y,vp[i].pt.x);
			if(inter==-1)
			{
				ULBSP::computeGrayscaleDescriptor8(roi,u,vp[i].pt.x,vp[i].pt.y,th,f[i]);
			}
			else
				ULBSP::computeGrayscaleDescriptor8(roi,inter,vp[i].pt.x,vp[i].pt.y,th,f[i]);
		}
		gx = f[1]-f[0] + 4*(f[3]-f[2]) + f[5]-f[4];
		return gx;
	}

	int yEdgeMag(Mat & src, int type=0, int inter=-1){
		//Rect rroi(1,1,src.cols-1,src.rows-1);
		int gy;
		const Mat_<uchar> roi=src;
		unsigned long int th=5;
		ushort f[9];
		vector<KeyPoint> vp;
		if(type==0)
			vp={KeyPoint(2,2,5),KeyPoint(3,2,5),KeyPoint(4,2,5),KeyPoint(2,4,5),KeyPoint(3,4,5),KeyPoint(4,4,5)};
		else
			vp={KeyPoint(2,2,5),KeyPoint(4,2,5),KeyPoint(6,2,5),KeyPoint(2,6,5),KeyPoint(4,6,5),KeyPoint(6,6,5)};
		//vector<KeyPoint> vp(&ar[0], &ar[0]+6);

		for (uint i=0; i<vp.size(); i++) {
			if(inter==-1)
			{
				ULBSP::computeGrayscaleDescriptor8(roi,roi.at<uchar>(vp[i].pt.y,vp[i].pt.x),vp[i].pt.x,vp[i].pt.y,th,f[i]);
			}
			else
				ULBSP::computeGrayscaleDescriptor8(roi,inter,vp[i].pt.x,vp[i].pt.y,th,f[i]);


		}
		gy = f[3]-f[0] + 4*(f[4]-f[1]) + f[5]-f[2];
		return gy;
	}

	vector<KeyPoint> mat2Keypoints(Mat & mat, int kpsize) {

	    vector<KeyPoint>  c_keypoints;

	    for ( int i = 0; i < mat.rows; i++) {
	    	 for ( int j = 0; j < mat.cols; j++) {
				KeyPoint kp(Point2f(j,i),kpsize);
				c_keypoints.push_back(kp);
	    	 }
	    }
	    return c_keypoints;
	}
};


#endif /* SOBEL_H_ */
