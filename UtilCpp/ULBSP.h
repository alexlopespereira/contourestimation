#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "DistanceUtils.h"

//#define SIZE_MAX 18446744073709551615UL
/*!
	Local Binary Similarity Pattern (LBSP) feature extractor

	Note 1: both grayscale and RGB/BGR images may be used with this extractor.
	Note 2: using LBSP::compute2(...) is logically equivalent to using LBSP::compute(...) followed by LBSP::reshapeDesc(...).

	For more details on the different parameters, see G.-A. Bilodeau et al, "Change Detection in Feature Space Using Local
	Binary Similarity Patterns", in CRV 2013.

	This algorithm is currently NOT thread-safe.
 */
class ULBSP  {
public:
	//! constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
	ULBSP(){}
	//! default destructor

	inline static uchar getVal(const uchar * data, int step, int y, int x, int _y, int _x)
	{
		uchar u = data[step*(_y+y)+_x+x];
		return u;
	}

	inline static uchar getVal(const uchar * data, int step, int x, int y, int _x, int _y, int n)
	{
		//uchar u = data[step*(_y+y)+_x+x];
		uchar u =data[step*(_y+y)+3*(_x+x)+n];
		return u;
	}

	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (1-channel version)
	inline static void computeGrayscaleDescriptor8(const cv::Mat& oInputImg, const uchar _ref, const int _x, const int _y, const size_t _t, ushort& _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC1);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_8bits_dbcross_1ch.i"
	}
	// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (3-channels version)
	inline static void computeRGBDescriptor(const cv::Mat& oInputImg, const uchar* const _ref,  const int _x, const int _y, const size_t _t, ushort* _res) {
		CV_DbgAssert(!oInputImg.empty());
		CV_DbgAssert(oInputImg.type()==CV_8UC3);
		CV_DbgAssert(LBSP::DESC_SIZE==2); // @@@ also relies on a constant desc size
		CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
		CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
		const size_t _step_row = oInputImg.step.p[0];
		const uchar* const _data = oInputImg.data;
		#include "LBSP_16bits_dbcross_3ch1t.i"
//#ifdef _val
//#error "definitions clash detected"
//#else
//#define _val(x,y,n) _data[_step_row*(_y+y)+3*(_x+x)+n]
//#endif
//
//for(int n=0; n<3; ++n) {
//	_res[n] = ((absdiff_uchar(getVal(_data,_step_row,-1, 1,_x,_y, n),_ref[n]) > _t) << 15)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 1,-1,_x,_y, n),_ref[n]) > _t) << 14)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 1, 1,_x,_y, n),_ref[n]) > _t) << 13)
//			+ ((absdiff_uchar(getVal(_data,_step_row,-1,-1,_x,_y, n),_ref[n]) > _t) << 12)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 1, 0,_x,_y, n),_ref[n]) > _t) << 11)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 0,-1,_x,_y, n),_ref[n]) > _t) << 10)
//			+ ((absdiff_uchar(getVal(_data,_step_row,-1, 0,_x,_y, n),_ref[n]) > _t) << 9)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 0, 1,_x,_y, n),_ref[n]) > _t) << 8)
//			+ ((absdiff_uchar(getVal(_data,_step_row,-2,-2,_x,_y, n),_ref[n]) > _t) << 7)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 2, 2,_x,_y, n),_ref[n]) > _t) << 6)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 2,-2,_x,_y, n),_ref[n]) > _t) << 5)
//			+ ((absdiff_uchar(getVal(_data,_step_row,-2, 2,_x,_y, n),_ref[n]) > _t) << 4)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 0, 2,_x,_y, n),_ref[n]) > _t) << 3)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 0,-2,_x,_y, n),_ref[n]) > _t) << 2)
//			+ ((absdiff_uchar(getVal(_data,_step_row, 2, 0,_x,_y, n),_ref[n]) > _t) << 1)
//			+ ((absdiff_uchar(getVal(_data,_step_row,-2, 0,_x,_y, n),_ref[n]) > _t));
//}
//
//#undef _val
	}
	//! utility, specifies the pixel size of the pattern used (width and height)
	static const size_t PATCH_SIZE = 5;

//protected:
	//! classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
	//virtual void computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;

};
