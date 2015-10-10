// note: this is the LBSP 16 bit double-cross single channel pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//    		 
//    O O O           ..  7  0  5 ..
//    O X O      =>   ..  1  X  3 ..
//    O O O           ..  4  2  6 ..

//
// must be defined externally:
//		_t				(size_t, absolute threshold used for comparisons)
//		_ref			(uchar, 'central' value used for comparisons)
//		_data			(uchar*, single-channel data to be covered by the pattern)
//		_y				(int, pattern rows location in the image data)
//		_x				(int, pattern cols location in the image data)
//		_step_row		(size_t, step size between rows, including padding)
//		_res			(ushort, 16 bit result vector)
//		absdiff_uchar	(function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y) _data[_step_row*(_y+y)+_x+x]
#endif

//    O O O           .. 15  8 13 ..
//  O O X O O    =>       9  X 11  
//    O O O           .. 12 10 14 ..

_res= ((absdiff_uchar(_val(-1,-1),_ref) > _t) << 7)
	+ ((absdiff_uchar(_val( 1, 1),_ref) > _t) << 6)
	+ ((absdiff_uchar(_val(-1, 1),_ref) > _t) << 5)
	+ ((absdiff_uchar(_val( 1,-1),_ref) > _t) << 4)
	+ ((absdiff_uchar(_val( 0, 1),_ref) > _t) << 3)
	+ ((absdiff_uchar(_val( 1, 0),_ref) > _t) << 2)
    + ((absdiff_uchar(_val( 0,-1),_ref) > _t) << 1)
    + ((absdiff_uchar(_val(-1, 0),_ref) > _t) << 0);
	
	
	

#undef _val
		