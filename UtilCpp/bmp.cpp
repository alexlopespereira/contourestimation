

 extern "C" {
//#include <stdio.h>
//#include <math.h>

}

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include "UtilCpp.h"
#include "OOTestCase.h"
#include "ULBSP.h"
//#include "cvWrapLEO.h"
#include "susan.h"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_ieee_utils.h>
#include <unordered_set>


 using namespace Util;
//using namespace OOMOV;
using namespace cv;

extern "C" {
void getSusanEdges(unsigned char * buffer, unsigned char * out,int t, int x_size, int y_size);
void getSusanPrincipal(unsigned char * buffer, int t, int x_size, int y_size);
}

static int countFrames=0;
 unsigned char rgbglob[H][W][3];

 /* number of bins of HSV in histogram */
 #define NH 10
 #define NS 10
 #define NV 10
 #define NVV 128

 /* max HSV values */
 #define H_MAX 360.0
 #define S_MAX 1.0
 #define V_MAX 1.0

 /* low thresholds on saturation and value for histogramming */
 #define S_THRESH 0.1
 #define V_THRESH 0.2

float Util::UtilCpp::prob12[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

void UtilCpp::bmpHeader(ofstream & ofile, int h, int w, int bpp)
{
	header hdr;
	char Type[2];
	Type[0]='B';
	Type[1]='M';
	hdr.Size=h*w*bpp/8+54;
	hdr.Reserved=0;
	hdr.OffSet = 54;
	hdr.headsize=40;
	hdr.Width = w;
	hdr.Height = h;
	hdr.Planes =1;
	hdr.BitsPerPixel=bpp;
	hdr.Compression=0;
	hdr.SizeImage=0;//h*w*bps/8;
	hdr.XPixelsPreMeter=3779;
	hdr.YPixelsPreMeter=3779;
	hdr.ColorsUsed=0;
	hdr.ColorsImportant=0;

//	fwrite(Type, sizeof(char), 2, f);
//	fwrite(&hdr, sizeof(char), sizeof (hdr), f);

	ofile.write( Type, 2 );
	ofile.write((const char *) &hdr, sizeof (hdr) );
}

void UtilCpp::videoSetup(Mat firstImg, string file)
{
	  videoOutput = new VideoWriter(file.c_str(), CV_FOURCC('P','I','M','1'), 25, firstImg.size(), true);
	  if(!videoOutput->isOpened()){
		std::cout << "error: could not open video file" << std::endl;
		exit(0);
	  }

}

void UtilCpp::addFrameToVideo(Mat img)
{
		(*videoOutput) << img; // send to video writer object
}

void UtilCpp::finishVideoWriting()
{
	delete videoOutput;
}

void UtilCpp::printStatistcs(int fit, int wh, string & str)
{
cout << str << endl;
cout << "N. of shadow pixels: " << fit << endl;
cout << "Image Size: " << wh << endl;
cout << "\% fit = "<< setprecision( 5 ) << 100 * fit / wh << "\%" << endl;
}


unsigned char UtilCpp::readRGB(const char * path, unsigned char * img, int w, int h, int bpp)
{

	   FILE *arq_bmp;
	   int inic_imagem=0;

	   if ((arq_bmp = fopen(path, "rb"))==NULL) {
	     printf("\nFile not Found: %s", path);
	     return 0;
	   }

	   fseek(arq_bmp,10,0);
	   fread(&inic_imagem, 1, sizeof(inic_imagem), arq_bmp);
	   //cout << inic_imagem << endl;
	   fseek(arq_bmp,inic_imagem,0);
	   fread(img,1,w*h*bpp/8,arq_bmp);

	   fclose(arq_bmp);

	      return 1;
}

unsigned char UtilCpp::readRGB(string & path, unsigned char * img, int w, int h, int bpp)
{
	//ifstream::pos_type size;
	int offset;

	ifstream file;
	file.open(path.c_str(), ios::in|ios::binary);
	if (file.is_open())
	{
		//size = file.tellg();
		file.seekg (10, ios::beg);
		offset = file.get();
		file.seekg( offset, ios::beg );
		file.read ( (char *)img, w*h*bpp/8);
		file.close();
	}
	else cout << "Unable to open file";

	      return 1;
}

void UtilCpp::writeRGBWithHeader(char * path, const char * headerpath, unsigned char *img, int w, int h, int bpp)
{
	unsigned char header[1078];
	FILE *arq_bmp;
   	if ((arq_bmp = fopen(headerpath, "rb"))==NULL) {
   		fprintf(stderr, "\nFile not Found: %s", headerpath);
     	return;
   	}
   	fread(header,1,1078,arq_bmp);
   	fclose(arq_bmp);

	FILE *fout;
	fout = fopen(path,"wb");
	if (fout==NULL) {
		fprintf(stderr, "\nFile not Found: %s", path);
    	return;
   	}
	fwrite(header, sizeof(char), 1078, fout);
	fwrite(img, sizeof(char), w*h*bpp/8, fout);
   	fclose(fout);
}

void UtilCpp::writeRGBWithHeader(string & path, string & hdr, unsigned char *img, int w, int h, int bpp)
{
//	const char header160x128[1078]={0x42,0x4D,0x36,0xF4,0x00,0x00,0x00,0x00,0x00,0x00,0x36,0x04,0x00,0x00,0x28,0x00,0x00,0x00,0xA0,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x01,0x00,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0xF0,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xCD,0x01,0x01,0x01,0xCD,0x02,0x02,0x02,0xCD,0x03,0x03,0x03,0xCD,0x04,0x04,0x04,0xCD,0x05,0x05,0x05,0xCD,0x06,0x06,0x06,0xCD,0x07,0x07,0x07,0xCD,0x08,0x08,0x08,0xCD,0x09,0x09,0x09,0xCD,0x0A,0x0A,0x0A,0xCD,0x0B,0x0B,0x0B,0xCD,0x0C,0x0C,0x0C,0xCD,0x0D,0x0D,0x0D,0xCD,0x0E,0x0E,0x0E,0xCD,0x0F,0x0F,0x0F,0xCD,0x10,0x10,0x10,0xCD,0x11,0x11,0x11,0xCD,0x12,0x12,0x12,0xCD,0x13,0x13,0x13,0xCD,0x14,0x14,0x14,0xCD,0x15,0x15,0x15,0xCD,0x16,0x16,0x16,0xCD,0x17,0x17,0x17,0xCD,0x18,0x18,0x18,0xCD,0x19,0x19,0x19,0xCD,0x1A,0x1A,0x1A,0xCD,0x1B,0x1B,0x1B,0xCD,0x1C,0x1C,0x1C,0xCD,0x1D,0x1D,0x1D,0xCD,0x1E,0x1E,0x1E,0xCD,0x1F,0x1F,0x1F,0xCD,0x20,0x20,0x20,0xCD,0x21,0x21,0x21,0xCD,0x22,0x22,0x22,0xCD,0x23,0x23,0x23,0xCD,0x24,0x24,0x24,0xCD,0x25,0x25,0x25,0xCD,0x26,0x26,0x26,0xCD,0x27,0x27,0x27,0xCD,0x28,0x28,0x28,0xCD,0x29,0x29,0x29,0xCD,0x2A,0x2A,0x2A,0xCD,0x2B,0x2B,0x2B,0xCD,0x2C,0x2C,0x2C,0xCD,0x2D,0x2D,0x2D,0xCD,0x2E,0x2E,0x2E,0xCD,0x2F,0x2F,0x2F,0xCD,0x30,0x30,0x30,0xCD,0x31,0x31,0x31,0xCD,0x32,0x32,0x32,0xCD,0x33,0x33,0x33,0xCD,0x34,0x34,0x34,0xCD,0x35,0x35,0x35,0xCD,0x36,0x36,0x36,0xCD,0x37,0x37,0x37,0xCD,0x38,0x38,0x38,0xCD,0x39,0x39,0x39,0xCD,0x3A,0x3A,0x3A,0xCD,0x3B,0x3B,0x3B,0xCD,0x3C,0x3C,0x3C,0xCD,0x3D,0x3D,0x3D,0xCD,0x3E,0x3E,0x3E,0xCD,0x3F,0x3F,0x3F,0xCD,0x40,0x40,0x40,0xCD,0x41,0x41,0x41,0xCD,0x42,0x42,0x42,0xCD,0x43,0x43,0x43,0xCD,0x44,0x44,0x44,0xCD,0x45,0x45,0x45,0xCD,0x46,0x46,0x46,0xCD,0x47,0x47,0x47,0xCD,0x48,0x48,0x48,0xCD,0x49,0x49,0x49,0xCD,0x4A,0x4A,0x4A,0xCD,0x4B,0x4B,0x4B,0xCD,0x4C,0x4C,0x4C,0xCD,0x4D,0x4D,0x4D,0xCD,0x4E,0x4E,0x4E,0xCD,0x4F,0x4F,0x4F,0xCD,0x50,0x50,0x50,0xCD,0x51,0x51,0x51,0xCD,0x52,0x52,0x52,0xCD,0x53,0x53,0x53,0xCD,0x54,0x54,0x54,0xCD,0x55,0x55,0x55,0xCD,0x56,0x56,0x56,0xCD,0x57,0x57,0x57,0xCD,0x58,0x58,0x58,0xCD,0x59,0x59,0x59,0xCD,0x5A,0x5A,0x5A,0xCD,0x5B,0x5B,0x5B,0xCD,0x5C,0x5C,0x5C,0xCD,0x5D,0x5D,0x5D,0xCD,0x5E,0x5E,0x5E,0xCD,0x5F,0x5F,0x5F,0xCD,0x60,0x60,0x60,0xCD,0x61,0x61,0x61,0xCD,0x62,0x62,0x62,0xCD,0x63,0x63,0x63,0xCD,0x64,0x64,0x64,0xCD,0x65,0x65,0x65,0xCD,0x66,0x66,0x66,0xCD,0x67,0x67,0x67,0xCD,0x68,0x68,0x68,0xCD,0x69,0x69,0x69,0xCD,0x6A,0x6A,0x6A,0xCD,0x6B,0x6B,0x6B,0xCD,0x6C,0x6C,0x6C,0xCD,0x6D,0x6D,0x6D,0xCD,0x6E,0x6E,0x6E,0xCD,0x6F,0x6F,0x6F,0xCD,0x70,0x70,0x70,0xCD,0x71,0x71,0x71,0xCD,0x72,0x72,0x72,0xCD,0x73,0x73,0x73,0xCD,0x74,0x74,0x74,0xCD,0x75,0x75,0x75,0xCD,0x76,0x76,0x76,0xCD,0x77,0x77,0x77,0xCD,0x78,0x78,0x78,0xCD,0x79,0x79,0x79,0xCD,0x7A,0x7A,0x7A,0xCD,0x7B,0x7B,0x7B,0xCD,0x7C,0x7C,0x7C,0xCD,0x7D,0x7D,0x7D,0xCD,0x7E,0x7E,0x7E,0xCD,0x7F,0x7F,0x7F,0xCD,0x80,0x80,0x80,0xCD,0x81,0x81,0x81,0xCD,0x82,0x82,0x82,0xCD,0x83,0x83,0x83,0xCD,0x84,0x84,0x84,0xCD,0x85,0x85,0x85,0xCD,0x86,0x86,0x86,0xCD,0x87,0x87,0x87,0xCD,0x88,0x88,0x88,0xCD,0x89,0x89,0x89,0xCD,0x8A,0x8A,0x8A,0xCD,0x8B,0x8B,0x8B,0xCD,0x8C,0x8C,0x8C,0xCD,0x8D,0x8D,0x8D,0xCD,0x8E,0x8E,0x8E,0xCD,0x8F,0x8F,0x8F,0xCD,0x90,0x90,0x90,0xCD,0x91,0x91,0x91,0xCD,0x92,0x92,0x92,0xCD,0x93,0x93,0x93,0xCD,0x94,0x94,0x94,0xCD,0x95,0x95,0x95,0xCD,0x96,0x96,0x96,0xCD,0x97,0x97,0x97,0xCD,0x98,0x98,0x98,0xCD,0x99,0x99,0x99,0xCD,0x9A,0x9A,0x9A,0xCD,0x9B,0x9B,0x9B,0xCD,0x9C,0x9C,0x9C,0xCD,0x9D,0x9D,0x9D,0xCD,0x9E,0x9E,0x9E,0xCD,0x9F,0x9F,0x9F,0xCD,0xA0,0xA0,0xA0,0xCD,0xA1,0xA1,0xA1,0xCD,0xA2,0xA2,0xA2,0xCD,0xA3,0xA3,0xA3,0xCD,0xA4,0xA4,0xA4,0xCD,0xA5,0xA5,0xA5,0xCD,0xA6,0xA6,0xA6,0xCD,0xA7,0xA7,0xA7,0xCD,0xA8,0xA8,0xA8,0xCD,0xA9,0xA9,0xA9,0xCD,0xAA,0xAA,0xAA,0xCD,0xAB,0xAB,0xAB,0xCD,0xAC,0xAC,0xAC,0xCD,0xAD,0xAD,0xAD,0xCD,0xAE,0xAE,0xAE,0xCD,0xAF,0xAF,0xAF,0xCD,0xB0,0xB0,0xB0,0xCD,0xB1,0xB1,0xB1,0xCD,0xB2,0xB2,0xB2,0xCD,0xB3,0xB3,0xB3,0xCD,0xB4,0xB4,0xB4,0xCD,0xB5,0xB5,0xB5,0xCD,0xB6,0xB6,0xB6,0xCD,0xB7,0xB7,0xB7,0xCD,0xB8,0xB8,0xB8,0xCD,0xB9,0xB9,0xB9,0xCD,0xBA,0xBA,0xBA,0xCD,0xBB,0xBB,0xBB,0xCD,0xBC,0xBC,0xBC,0xCD,0xBD,0xBD,0xBD,0xCD,0xBE,0xBE,0xBE,0xCD,0xBF,0xBF,0xBF,0xCD,0xC0,0xC0,0xC0,0xCD,0xC1,0xC1,0xC1,0xCD,0xC2,0xC2,0xC2,0xCD,0xC3,0xC3,0xC3,0xCD,0xC4,0xC4,0xC4,0xCD,0xC5,0xC5,0xC5,0xCD,0xC6,0xC6,0xC6,0xCD,0xC7,0xC7,0xC7,0xCD,0xC8,0xC8,0xC8,0xCD,0xC9,0xC9,0xC9,0xCD,0xCA,0xCA,0xCA,0xCD,0xCB,0xCB,0xCB,0xCD,0xCC,0xCC,0xCC,0xCD,0xCD,0xCD,0xCD,0xCD,0xCE,0xCE,0xCE,0xCD,0xCF,0xCF,0xCF,0xCD,0xD0,0xD0,0xD0,0xCD,0xD1,0xD1,0xD1,0xCD,0xD2,0xD2,0xD2,0xCD,0xD3,0xD3,0xD3,0xCD,0xD4,0xD4,0xD4,0xCD,0xD5,0xD5,0xD5,0xCD,0xD6,0xD6,0xD6,0xCD,0xD7,0xD7,0xD7,0xCD,0xD8,0xD8,0xD8,0xCD,0xD9,0xD9,0xD9,0xCD,0xDA,0xDA,0xDA,0xCD,0xDB,0xDB,0xDB,0xCD,0xDC,0xDC,0xDC,0xCD,0xDD,0xDD,0xDD,0xCD,0xDE,0xDE,0xDE,0xCD,0xDF,0xDF,0xDF,0xCD,0xE0,0xE0,0xE0,0xCD,0xE1,0xE1,0xE1,0xCD,0xE2,0xE2,0xE2,0xCD,0xE3,0xE3,0xE3,0xCD,0xE4,0xE4,0xE4,0xCD,0xE5,0xE5,0xE5,0xCD,0xE6,0xE6,0xE6,0xCD,0xE7,0xE7,0xE7,0xCD,0xE8,0xE8,0xE8,0xCD,0xE9,0xE9,0xE9,0xCD,0xEA,0xEA,0xEA,0xCD,0xEB,0xEB,0xEB,0xCD,0xEC,0xEC,0xEC,0xCD,0xED,0xED,0xED,0xCD,0xEE,0xEE,0xEE,0xCD,0xEF,0xEF,0xEF,0xCD,0xF0,0xF0,0xF0,0xCD,0xF1,0xF1,0xF1,0xCD,0xF2,0xF2,0xF2,0xCD,0xF3,0xF3,0xF3,0xCD,0xF4,0xF4,0xF4,0xCD,0xF5,0xF5,0xF5,0xCD,0xF6,0xF6,0xF6,0xCD,0xF7,0xF7,0xF7,0xCD,0xF8,0xF8,0xF8,0xCD,0xF9,0xF9,0xF9,0xCD,0xFA,0xFA,0xFA,0xCD,0xFB,0xFB,0xFB,0xCD,0xFC,0xFC,0xFC,0xCD,0xFD,0xFD,0xFD,0xCD,0xFE,0xFE,0xFE,0xCD,0xFF,0xFF,0xFF,0xCD};
	//const char header320x240[1078] = {0x42,0x4D,0x36,0x84,0x03,0x00,0x00,0x00,0x00,0x00,0x36,0x00,0x00,0x00,0x28,0x00,0x00,0x00,0x40,0x01,0x00,0x00,0xF0,0x00,0x00,0x00,0x01,0x00,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x84,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xF3,0xFA,0xFF,0xBF,0xC6,0xCC,0x6F,0x7E,0x93,0x68,0x77,0x8C,0x6F,0x7E,0x93,0x68,0x77,0x8C,0x81,0x8B,0x97,0x60,0x6A,0x76,0x55,0x67,0x7B,0x50,0x62,0x76,0x59,0x60,0x6D,0x59,0x60,0x6D,0x45,0x4A,0x57,0x41,0x46,0x53,0x5C,0x62,0x72,0x62,0x68,0x78,0x7E,0x82,0x92,0x69,0x6D,0x7D,0x59,0x60,0x6D,0x59,0x60,0x6D,0x5C,0x62,0x72,0x62,0x68,0x78,0x79,0x8B,0xA1,0x74,0x86,0x9C,0x82,0x91,0xA6,0xCE,0xDD,0xF2,0x7F,0x85,0x8E,0x52,0x58,0x61,0x63,0x5D,0x58,0x58,0x52,0x4D,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x63,0x5E,0x64,0x67,0x62,0x68,0x62,0x5E,0x61,0x62,0x5E,0x61,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x62,0x5E,0x61,0x62,0x5E,0x61,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x62,0x5E,0x61,0x62,0x5E,0x61,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x62,0x5E,0x61,0x62,0x5E,0x61,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x62,0x5E,0x61,0x62,0x5E,0x61,0x63,0x60,0x61,0x62,0x5F,0x60,0x63,0x60,0x61,0x62,0x5F,0x60,0x63,0x5E,0x64,0x67,0x62,0x68,0x65,0x64,0x66,0x65,0x64,0x66,0x62,0x5E,0x61,0x62,0x5E,0x61,0x65,0x64,0x66,0x65,0x64,0x66,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x62,0x5E,0x61,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x68,0x63,0x64,0x68,0x63,0x64,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x63,0x60,0x61,0x62,0x5F,0x60,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x63,0x60,0x61,0x62,0x5F,0x60,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x5D,0x59,0x5C,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x63,0x60,0x61,0x62,0x5F,0x60,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x5B,0x57,0x5A,0x5B,0x57,0x5A,0x60,0x5C,0x5F,0x61,0x5D,0x60,0x63,0x60,0x61,0x62,0x5F,0x60,0x62,0x5E,0x61,0x62,0x5E,0x61,0x66,0x62,0x65,0x66,0x62,0x65,0x68,0x63,0x64,0x68,0x63,0x64,0x76,0x70,0x73,0x76,0x70,0x73,0xAC,0x9A,0x92,0xCC,0xBA,0xB2,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,0xFF,0xFF,0xF8,0xFF,0xFF,0xFE,0xFF,0xFF,0xF8,0xFF,0xFF,0xEE,0xFA,0xFF,0xE0,0xEC,0xFF,0xBC,0xC3,0xD7,0xA0,0xA7,0xBB,0xD4,0xDB,0xE1,0xFE,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
        char header[1078];
	ofstream::pos_type osize;
	int fsize;
	fsize=w*h*bpp/8;

        ifstream ifile;
	ifile.open(hdr.c_str(), ios::in|ios::binary);
	if (ifile.is_open())
	{
            ifile.read(header,1078);
            ifile.close();
        }

	ofstream ofile;
	ofile.open(path.c_str(), ios::out|ios::binary);
	if (ofile.is_open())
	{
		ofile.write( header, 1078 );
		ofile.write( (const char *)img, fsize );
		ofile.close();
	}
}

void UtilCpp::writeRGB(string & path, unsigned char *img, int w, int h, int bpp)
{
    int fsize;
    fsize=w*h*bpp/8;

    ofstream ofile;
    ofile.open(path.c_str(), ios::out|ios::binary);
    if (ofile.is_open())
    {
            bmpHeader(ofile, h, w, bpp);
            ofile.write( (const char *)img, fsize );
            ofile.close();
    }
}

void UtilCpp::writeRGB(string name, Mat & img, int n)
{
    char path[30];
    string p = name;
    if(n!=-1)
    {
		sprintf(path,"%04d.bmp",n);
		p = p+string(path);
    }
    else
    	p = p+".bmp";

    imwrite(p,img);
}

void UtilCpp::writeJPG(string name, Mat & img, int n)
{
    char path[30];
    if(n!=-1)
    	sprintf(path,"%04d.jpg",n);
    else
    	sprintf(path,".jpg");
    string p = name+string(path);
    imwrite(p,img);
}

void UtilCpp::writePNG(string name, Mat & img, int n)
{
    char path[30];
    if(n!=-1)
    	sprintf(path,"%04d.png",n);
    else
    	sprintf(path,".png");
    string p = name+string(path);
    imwrite(p,img);
}

void UtilCpp::writeImg(string name, Mat & img, int n, string ext)
{
    char path[30];
    sprintf(path,"%06d.",n);
    string p = name+string(path)+ext;
    imwrite(p,img);
}

void UtilCpp::writeRGB_OpenCv(string & path, unsigned char *img, int w, int h, int bpp)
{
	IplImage *image;
	image = cvCreateImage(cvSize(w,h), bpp/3, 3);
	image->imageData = (char *)img;
	if(!cvSaveImage(path.c_str(),image))
		printf("Could not save: %s\n",path.c_str());
	cvReleaseImage(&image);
}

void UtilCpp::writeBackground_OpenCv(int nfile, unsigned char * bg, int w, int h)
{
    char path[30];
    sprintf(path,"result/background%04d.bmp",nfile);
    string p(path);
    IplImage *image;
	image = cvCreateImage(cvSize(w,h), 8, 3);
	image->imageData = (char *)bg;
	if(!cvSaveImage(p.c_str(),image))
		printf("Could not save: %s\n",p.c_str());
//    writeRGB(p, bg, w, h, 24);
	cvReleaseImage(&image);
}

void UtilCpp::writeTest_OpenCv(int nfile, unsigned char * bg, int w, int h)
{
    char path[30];
    sprintf(path,"result/test%04d.bmp",nfile);
    string p(path);
    IplImage *image;
	image = cvCreateImage(cvSize(w,h), 8, 1);
	image->imageData = (char *)bg;
	if(!cvSaveImage(p.c_str(),image))
		printf("Could not save: %s\n",p.c_str());
//    writeRGB(p, bg, w, h, 24);
	cvReleaseImage(&image);
}

void UtilCpp::writeTest_OpenCv(int nfile, IplImage *image, int w, int h)
{
    char path[30];
    sprintf(path,"result/test%04d.bmp",nfile);
    string p(path);
	if(!cvSaveImage(p.c_str(),image))
		printf("Could not save: %s\n",p.c_str());
//    writeRGB(p, bg, w, h, 24);
//	cvReleaseImage(&image);
}

void UtilCpp::writeBackground(int nfile, Mat & img, int w, int h)
{
    char path[30];
    sprintf(path,"result/background%04d.bmp",nfile);
    string p(path);
    writeRGB(p, img.data, w, h, 24);
}

void UtilCpp::writeBackground(int nfile, Mat & img)
{
    char path[30];
    sprintf(path,"result/background%04d.bmp",nfile);
    string p(path);
//    writeRGB(p, bg, w, h, 24);
    imwrite(p,img);
}

void UtilCpp::convert1to255(unsigned char *src, unsigned char *dest, int len)
{
	int i;
	unsigned char c,d;
	for(i=0; i<len;i++)
	{
		c=*src++;
		d= c==1 ? 255:0;
		*dest++=d;
	}
}

void UtilCpp::writeResult(int nfile, unsigned char * img, const string * hdrpath, int w, int h, int bpp)
{
	char path[30];
	string p(*hdrpath);
	sprintf(path,"./result/output%04d.bmp",nfile);
//	string p2(path);
	string filepath(path);
	convert1to255(img,&irgb[0][0][0],w*h);
   	writeRGBWithHeader(path, p.c_str(), img, w, h, bpp);
//	writeRGB(p2, img, w, h, bpp);
}

void UtilCpp::writeResult(int nfile, unsigned char * img, int w, int h)
{
	char path[30];
	//char header[20];
	//strcpy(header, hdrpath);
   	sprintf(path,"result/output%04d.bmp",nfile);
   	string p(path);
   	int inc;
   	inc = BWtoRGBImage(img, &rgbglob[0][0][0], w, h);
//   	writeRGB(p, &rgb[0][0][0], w, h, bpp);
   	writeRGB(p, &rgbglob[0][0][0], w+inc, h, 24);
}

void UtilCpp::writeResult(int nfile, Mat &img)
{
	char path[30];
	//char header[20];
	//strcpy(header, hdrpath);
   	sprintf(path,"result/output%04d.bmp",nfile);
   	string p(path);
	imwrite(p, img);
}

void UtilCpp::writeResult(int nfile, Mat &img, string spath)
{
	char path[30];
   	sprintf(path,"%06d.jpg",nfile);
   	string p(path);
   	spath.append(p);
	imwrite(spath, img);
}

//void UtilCpp::countErrors(unsigned char *gt, unsigned char * pic, int *falsep, int *falsen, int *truep, int *truen, int w, int h, int bpp)
//{
//	int i, len=w*h*bpp/8;
//	*falsep=0;
//	*falsen=0;
//	unsigned char *temp, *temp1;
//	temp=gt;
//	temp1=pic;
//
//	for(i=0; i<len;i++)
//	{
//		if(*temp++<10)
//		{
//			if(*temp1++>=254)
//				(*falsep)++;
//			else
//				(*truen)++;
//		}
//		else
//		{
//			if(*temp1++<=10)
//				(*falsen)++;
//			else
//				(*truep)++;
//		}
//	}
//}



void UtilCpp::countErrors(Mat & gt, Mat & pic, cresult & r)
{
	int i, l, w, h;
	r.fn=0;
	r.fp=0;
	r.tp=0;
	r.tn=0;
	int gtfg=0, picfg=0;
	h=gt.rows;
	w=gt.cols;
	Mat_<uchar>& gtImg = (Mat_<uchar>&)gt;
	Mat_<uchar>& picImg = (Mat_<uchar>&)pic;
	for(i=0; i<h;i++)
	{
		for(l=0; l<w;l++)
		{
			if(gtImg(i,l)==BG)
			{
				if(picImg(i,l)==FG)
				{
					picfg++;
					r.fp++;
				}
				else if(picImg(i,l)==BG)
					r.tn++;
				else
					cerr << "unknown value: " << picImg(i,l) << endl;
			}
			else
			{
				gtfg++;
				if(picImg(i,l)==BG)
					r.fn++;
				else if(picImg(i,l)==FG)
				{
					picfg++;
					r.tp++;
				}
				else
					cerr << "unknown value: " << picImg(i,l) << endl;
			}
		}
	}
//	cout << gtfg << ", " << picfg << endl;
//	imshow("gt",gt);
//	imshow("pic", pic);
//	waitKey();
}


void UtilCpp::countErrorsShadowForeground(unsigned char *gt, unsigned char * imresult, float *ni, float *e, int w, int h, int bpp)
{
	int i, len=w*h, tps=0, fns=0, tpf=0, fnf=0;
	byte ir, ig, ib, gtr, gtg, gtb;
	unsigned char *temp, *temp1;
	temp=imresult;
	temp1=gt;
	
	for(i=0; i<len;i++)
	{
		ib=*temp++;
		ig=*temp++;
		ir=*temp++;

		gtb = *temp1++;
		gtg = *temp1++;
		gtr = *temp1++;

		if(gtr==FG || gtb == FG)
		{
			byte presult = comparePixelShadowForeground(ir, ig, ib, gtr, gtg, gtb);
			switch (presult)
			{
			case TRUE_PS:
				tps++;
				break;
			case FALSE_NS:
				fns++;
				break;
			case TRUE_PFG:
				tpf++;
				break;
			case FALSE_NFG:
				fnf++;
				break;
			default:
				//cout << " default " << endl;
				break;
			}
		}
	}
	*ni=tps/(float)(tps+fns);
	*e=tpf/(float)(tpf+fnf);
}

int UtilCpp::comparePixel(unsigned char gt, unsigned char ft)
{
	if(gt<=1)
	{
		if(ft>=254)
			return FALSE_P;
	}
	else
	{
		if(ft<=1)
			return FALSE_N;
	}
	return -1;
}

int UtilCpp::comparePixelShadowForeground(byte ir, byte ig, byte ib, byte gtr, byte gtg, byte gtb)
{
	if(gtr == FG && gtg ==0 && gtb == 0)
	{
		if(ir == FG && ig ==0 && ib == 0)
			return TRUE_PS;
		else if(ir == 0 && ig ==0 && ib == FG)
			return FALSE_NS;
	}
	else if(gtr == 0 && gtg ==0 && gtb == FG)
	{
		if(ir == 0 && ig ==0 && ib == FG)
			return TRUE_PFG;
		else if(ir == FG && ig ==0 && ib == 0)
			return FALSE_NFG;
	}
//	else
//	{
//		if(ir == 0 && ig ==0 && ib == FG)
//			return FALSE_PFG;
//		else if(ir == FG && ig ==0 && ib == 0)
//			return FALSE_PS;
//	}
	return -1;
}

void UtilCpp::writeDifference(unsigned char * im, unsigned char *gt, unsigned short h, unsigned short w, int nfile)
	{

		int i,j, r;
		unsigned char i1,i2, *iptr;
		UtilCpp u;
		unsigned char imr[h][w][3];
		iptr=&imr[0][0][0];
		for(i=0; i<h; i++)
		{
			for(j=0; j<w; j++)
			{
				i1=*im++;
				i2=*gt++;
				r=u.comparePixel(i2, i1);
				if(r==FALSE_N)
				{
					*iptr++=FG;
					*iptr++=0;
					*iptr++=0;
				}
				else if(r==FALSE_P)
				{
					*iptr++=0;
					*iptr++=0;
					*iptr++=FG;
				}
				else
				{
					*iptr++=i1;
					*iptr++=i1;
					*iptr++=i1;
				}
			}
		}

		char path[30];
	   	sprintf(path,"compare/output%04d.bmp",nfile);
	   	string p(path);
	   	writeRGB(p, &imr[0][0][0], w, h, 24);
	}

void UtilCpp::RGBtoBWImage(unsigned char *p, unsigned char *bw, int bwlen)
{
	int i;
	unsigned char c[3], result;
	for(i=0; i<bwlen; i++)
	{
		c[0]=(unsigned char) *p++;
		c[1]=(unsigned char) *p++;
		c[2]=(unsigned char) *p++;
		result=MAX(c[0], c[1]);
		*bw = MAX( result, c[2]);
		bw++;
	}

}

void UtilCpp::RGBtoBWImage1(unsigned char *p, unsigned char *bw, int bwlen)
{
	int i;
	unsigned char c[3], result;
	for(i=0; i<bwlen; i++)
	{

		c[0]=(unsigned char) *p++;
		c[1]=(unsigned char) *p++;
		c[2]=(unsigned char) *p++;
		if(c[0]>0 || c[1]>0 || c[2]>0)
			*bw++=1;
		else
			*bw++=0;
	}

}
void UtilCpp::changeColor(byte * in, byte *out, int c, int th, int len, bool eq)
{
	int i;
	byte ir, ig, ib, *temp;
	byte to=FG;

	temp=out;
	for(i=0; i<len;i++)
	{
		ib=*in++;
		ig=*in++;
		ir=*in++;

		if(ir + ig + ib > th)
		{
			if(c==1)
			{
				*temp=to;
				temp+=3;
			}
			else if(c==2)
			{
				temp++;
				*temp=to;
				temp+=2;
			}
			else if(c==3)
			{
				temp+=2;
				*temp=to;
				temp++;
			}
		}
		else
		{
			temp+=3;
//			if(*out==FG) { out+=3; continue;}
//			*out++=0;
//			*out++=0;
//			if(*out==FG) { out+=1; continue;}
//			*out++=0;
		}
	}
}

int UtilCpp::BWtoRGBImage(unsigned char *bw, unsigned char *rgb, int w, int h)
{
	int i,j;
	unsigned char c,d;
	int mod;
	mod=(w%4);
	if(mod)
		mod=4-mod;

	for(i=0; i<h; i++)
	{
		for(j=0; j<w; j++)
		{
			c=(unsigned char) *bw++;
			d = c==1 ? 255:0;
			*rgb++ = d;
			*rgb++ = d;
			*rgb++ = d;
		}
		for(j=0; j<mod; j++)
		{
			*rgb++ = 0;
			*rgb++ = 0;
			*rgb++ = 0;
		}
	}
	return mod;
}

void UtilCpp::RGBtoHSV( unsigned char cr, unsigned char cg, unsigned char cb, float *h, float *s, float *v )
{
	double min, max, delta, r, g, b;

	min = MIN( MIN(cr, cg), cb )/255.0;
	max = MAX( MAX(cr, cg), cb )/255.0;
	r=cr/255.0;
	g=cg/255.0;
	b=cb/255.0;

	*v = rint(max*100);				// v

	delta = max - min;

	if( max != 0 )
		*s = 100*delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		*s = 0;
		*h = 0;
		*v = 0;
		return;
	}

	if(max == min)
		*h = 0;
	else if( r == max )
		if(g>=b)
			*h = 60 * ( g - b ) / delta;		// between yellow & magenta
		else
			*h = 360 + 60 * ( g - b ) / delta;
	else if( g == max )
		*h = 120 + 60 * ( b - r ) / delta;	// between cyan & yellow
	else
		*h = 240 + 60 * ( r - g ) / delta;	// between magenta & cyan

	return;
}

void UtilCpp::YUVtoRGB(unsigned char y, unsigned char cb, unsigned char cr, unsigned char *r, unsigned char *g, unsigned char *b )
{
	*r = (unsigned char)( 298.082 * y                + 408.583 * cr ) / 256 - 222.921;
	*g = (unsigned char)( 298.082 * y - 100.291 * cb - 208.120 * cr ) / 256 + 135.576;
	*b = (unsigned char)( 298.082 * y + 516.412 * cb                ) / 256 - 276.836;

}


char *  UtilCpp::createFilename(char * path, char * filename, int number, int characters, string & ext)
{
	if(characters==6)
		sprintf(filename, "%06d.", number);
	else if(characters==5)
		sprintf(filename, "%05d.", number);
	else if(characters==4)
		sprintf(filename, "%04d.", number);
	else if(characters==3)
		sprintf(filename, "%03d.", number);
	else if(characters==1)
		sprintf(filename, "%01d.", number);
	else
		fprintf(stderr, "\nWrong number of characters to create file name.");

	strcat(path, filename);
	strcat(path, ext.c_str());
	return filename;
}

string UtilCpp::createFilename(string & path, int number, int characters, string & ext)
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
void  UtilCpp::reportArray(unsigned short * array, int l, int c, char * filename)
{
//	FILE *f;
//	int i, j;
//	f= fopen(filename, "w");
//	for(i=0; i<l; i++)
//	{
//		for(j=0;j<c;j++)
//		{
//			fprintf(f, "%hd,", *array++);
//		}
//		fprintf(f, "\n");
//	}
//	fclose(f);
}




float UtilCpp::probVerifyConnectivity8(unsigned char * p, int w)
{
	unsigned char *e, *ec, *c, *dc;
	int count=0;
	c=p-w;
	e=p-1;
	dc=c+1;
	ec=c-1;

	if(*c==FG )
		count++;

	if(*e==FG)
		count++;

	if(*dc==FG)
		count++;

	if(*ec==FG)
		count++;

	return prob12[count];
}

float UtilCpp::probVerifyConnectivity16(unsigned char * p, int w, float *prob8)
{
	unsigned char *e, *ec, *c, *dc;
	short count=0;
	c=p-w;
	e=p-1;
	dc=c+1;
	ec=c-1;
	if(*c==FG )
		count++;

	if(*e==FG)
		count++;

	if(*dc==FG)
		count++;

	if(*ec==FG)
		count++;

	*prob8 = prob12[count];

	if(*(c-w)==FG )
		count++;

	if(*(e-1)==FG)
		count++;

	if(*(dc-w)==FG)
		count++;

	if(*(ec-w)==FG)
		count++;

	if(*(ec-1)==FG)
		count++;

	if(*(ec-w-1)==FG)
		count++;


	return prob12[count];
}

float UtilCpp::probVerifyConnectivityAfter(unsigned char * p, int w)
{
	unsigned char *b, *be, *bd, *d;
	int count=0;
	b=p+w;
	d=p+1;
	be=b-1;
	bd=b+1;

	if(*d==FG)
		count++;

	if(*b==FG)
		count++;

	if(*be==FG)
		count++;

	if(*bd==FG)
		count++;

	return prob12[count];
}


int UtilCpp::verifyConnectivityAfter(unsigned char * p, int w)
{
	unsigned char *b, *be, *bd, *d;
	int count=0;
	b=p+w;
	d=p+1;
	be=b-1;
	bd=b+1;

	if(*d==FG)
		count++;

	if(*b==FG)
		count++;

	if(*be==FG)
		count++;

	if(*bd==FG)
		count++;

	return count;
}

float UtilCpp::verifyColor(unsigned char *p, int w, unsigned char wcolor)
{
	unsigned char *e, *ec, *c, *dc, *b, *be, *bd, *d;
	int rgbw=3*w;
	float rm, gm, bm, result, mean;
	unsigned char pr, pg, pb;
	b=p+rgbw;
	d=p+1;
	be=b-1;
	bd=b+1;
	c=p-rgbw;
	e=p-1;
	dc=c+1;
	ec=c-1;

	pr=*p++;
	pg=*p++;
	pb=*p++;

	rm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))/8;
	gm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))/8;
	bm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))/8;

	mean = (abs(rm-pr) + abs(gm-pg) + abs(bm-pb));
	if(mean == 0)
		result = wcolor;
	else
		result = wcolor/mean;

	return result;

}


short UtilCpp::verifySimilarFGColor(unsigned char *p, int w)
{
	unsigned char *e, *ec, *c, *dc, *b, *be, *bd, *d;
	int rgbw=3*w;
	int rm, gm, bm, result;
	unsigned char pr, pg, pb;
	b=p+rgbw;
	d=p+1;
	be=b-1;
	bd=b+1;
	c=p-rgbw;
	e=p-1;
	dc=c+1;
	ec=c-1;

	pr=*p++;
	pg=*p++;
	pb=*p++;

	rm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))>>3;
	gm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))>>3;
	bm=((*b++) + (*d++) + (*be++) + (*bd++) + (*c++) + (*e++) + (*dc++) + (*ec++))>>3;

	result = (abs(rm-pr) + abs(gm-pg) + abs(bm-pb));

	return result;

}

short UtilCpp::verifySimilarFGColor(Mat & img, int y, int x)
{
	short result;
	int i,j;
	Rect roi(x-1,y-1,3,3);
	Rect imgRec(0,0,img.cols,img.rows);
	Rect correctRoi = imgRec & roi;
//	if(correctRoi.width!=roi.width || correctRoi.height!=roi.height)
//		cout << "dif" << endl;

	Mat roiImg = img(correctRoi);
	Mat_<Vec3b>& vroiImg = (Mat_<Vec3b>&)roiImg;
	Vec3i rl,center;
	rl = Vec3i(0,0,0);

	for(i=0; i<correctRoi.height; i++)
	{
		for(j=0; j<correctRoi.width; j++)
		{
			rl+=vroiImg(i,j);
		}
	}

	if(x==0||y==0)
		center=vroiImg(0,0);
	else
		center=vroiImg(1,1);
	rl=(rl-center);
	int div=(correctRoi.width*correctRoi.height);
	result = abs(rl[0]/div-center[0])+abs(rl[1]/div-center[1])+abs(rl[2]/div-center[2]);
	return result;
}

/*short UtilCpp::verifyConnectivity(Mat & src, int y, int x)
{
	short count=0;
	int i,j;
	Rect roi(x-2,y-2,5,5);
	Rect imgRec(0,0,src.cols,src.rows);
	Rect correctRoi = imgRec & roi;
	Mat roiImg = src(correctRoi);
	Mat_<uchar>& vroiImg = (Mat_<uchar>&)roiImg;
	short leny, lenx;
	leny=y-correctRoi.y;
	lenx=x-correctRoi.x;

	for(i=0; i<leny; i++)
	{
		for(j=0; j<correctRoi.width; j++)
		{
			if(vroiImg(i,j)==FG)
				count++;
			else if(vroiImg(i,j)==BG)
				count=count;
			else
				cout << "unknown value" << vroiImg(i,j) << endl;
		}
	}
	if(lenx>1)
	{
		if(vroiImg(leny,lenx-2)==FG)
			count++;
	}
	if(lenx>0)
	{
		if(vroiImg(leny,lenx-1)==FG)
			count++;
	}

	return count;
}*/

short UtilCpp::verifyConnectivity(unsigned char * p, int w)
{
	unsigned char *e, *ec, *c, *dc;
	short count=0;
	c=p-w;
	e=p-1;
	dc=c+1;
	ec=c-1;
	if(*c==FG )
		count++;

	if(*e==FG)
		count++;

	if(*dc==FG)
		count++;

	if(*ec==FG)
		count++;

	if(*(c-w)==FG )
		count++;

	if(*(e-1)==FG)
		count++;

	if(*(dc-w)==FG)
		count++;

	if(*(ec-w)==FG)
		count++;

	if(*(ec-1)==FG)
		count++;

	if(*(ec-w-1)==FG)
		count++;
//	if(count>6)
//		cout << endl << "count " << count << endl;
	return count;
}

void UtilCpp::setProb(float p0, float p1, float p2, float p3, float p4)
{
	prob12[0]=p0;
	prob12[1]=p1;
	prob12[2]=p2;
	prob12[3]=p3;
	prob12[4]=p4;
}

void UtilCpp::setProb16(float p0, float p1, float p2, float p3, float p4, float p5, float p6, float p7, float p8, float p9, float p10, float p11)
{
	prob12[0]=p0;
	prob12[1]=p1;
	prob12[2]=p2;
	prob12[3]=p3;
	prob12[4]=p4;
	setAfterProb(p5, p6, p7, p8);
	prob12[9]=p9;
	prob12[10]=p10;
	prob12[11]=p11;
}
void UtilCpp::setAfterProb(float p5, float p6, float p7, float p8)
{
	prob12[5]=p5;
	prob12[6]=p6;
	prob12[7]=p7;
	prob12[8]=p8;
}

int UtilCpp::compareImages(Mat & img1, Mat & img2, int w, int h, bool print)
{
	int i, j, count=0;
	unsigned char r1, r2, g1, g2, b1, b2;
	Mat_<uchar>& uimg1 = (Mat_<uchar>&)img1;
	Mat_<uchar>& uimg2 = (Mat_<uchar>&)img2;

	for(i=0; i<h; i++)
	{
		for(j=0; j<w; j++)
		{
			r1=uimg1(i,j);
//			g1=*img1++;
//			b1=*img1++;
			r2=uimg2(i,j);
//			g2=*img2++;
//			b2=*img2++;

			if(r1!=r2)
			{
				if(print)
				{
					cout << "r1=" << (int)r1 << ", r2=" << (int)r2 << " in (i,j)=" << i << "," << j << endl;
				}
				count++;
			}
//			if(g1!=g2)
//			{
//				if(print)
//					cout << "g1=" << (int)g1 << ", g2=" << (int)g2 << " in (i,j)=" << i << "," << j << endl;
//				count++;
//			}
//			if(b1!=b2)
//			{
//				if(print)
//					cout << "b1=" << b1 << ", b2=" << b2 << " in (i,j)=" << i << "," << j << endl;
//				count++;
//			}
		}
	}
	return count;
}

float UtilCpp::evPoly(float * coefs, unsigned char order, int x)
{
	int i;
	float y = *coefs++;
	float f = x;

    for (i = 0; i < order; i++)
    {
        y += (*coefs++) * f;
        f *= x;
    }
    return y;
}

void UtilCpp::drawSquare(unsigned char * img, unsigned short w, square *sqr, unsigned char v)
{
	int i;
	unsigned char *lup, *ldow, *rleft, *rright;
	lup=img+sqr->y1*w+sqr->x1;
	ldow=img+sqr->y2*w+sqr->x1;
	for(i=sqr->x1; i<=sqr->x2; i++)
	{
		*lup++=v;
		*ldow++=v;
	}

	rleft=img+sqr->y1*w+sqr->x1;
	rright=img+sqr->y1*w+sqr->x2;
	for(i=sqr->y1; i<=sqr->y2; i++)
	{
		*rleft=v;
		*rright=v;

		rleft+=w;
		rright+=w;
	}
}

//void UtilCpp::drawRGBSquare(unsigned char * img, unsigned short w,  unsigned short h, square *sqr, unsigned short objID, bool writeNumber)
//{
//	int i;
//	unsigned char *lup, *ldow, *rleft, *rright;
//	RGB *c;
//	c=&bbcolors[objID%12];
//	lup=img+(sqr->y1*w+sqr->x1)*3;
//	ldow=img+(sqr->y2*w+sqr->x1)*3;
//	for(i=sqr->x1; i<=MIN(sqr->x2,w-1); i++)
//	{
//		*lup++=c->R;
//		*lup++=c->G;
//		*lup++=c->B;
//		*ldow++=c->R;
//		*ldow++=c->G;
//		*ldow++=c->B;
//	}
//
//	rleft=img+3*(sqr->y1*w+sqr->x1);
//	rright=img+3*(sqr->y1*w+sqr->x2);
//	for(i=sqr->y1; i<=MIN(sqr->y2,h-1); i++)
//	{
//		*(rleft)=c->R;
//		*(rleft+1)=c->G;
//		*(rleft+2)=c->B;
//		*rright=c->R;
//		*(rright+1)=c->G;
//		*(rright+2)=c->B;
//
//		rleft+=w*3;
//		rright+=w*3;
//	}
//
//	if(writeNumber)
//	{
//		char path[30];
//		Point point(sqr->x1, sqr->y2);
//		IplImage *image =  cvCreateImageHeader(cv::Size(w, h),IPL_DEPTH_8U,3);
//		image->imageData = (char *)img;
//		sprintf(path,"%03d",objID);
//		drawText(path, image, &point);
//		cvReleaseImageHeader(&image);
//	}
//
//}

void UtilCpp::drawText(char * txt, IplImage* image, Point *orig)
{
	// Use "y" to show that the baseLine is about
	string text(txt);// = "Funny text inside the box";
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	Mat img(image);
	int baseline=0;
	Size textSize = cv::getTextSize(text, fontFace,fontScale, thickness, &baseline);
	baseline += thickness;
	// center the text
//	Point orig(0,image->height/2);
	// then put the text itself
	putText(img, text, *orig, fontFace, fontScale,
	Scalar::all(255), thickness, 8);
	char outFileName[] = "./test.bmp";
}

bool UtilCpp::compareSquares(square *s1, square *s2)
{
	if(s1->x1==s2->x1 && s1->x2==s2->x2 && s1->y1==s2->y1 && s1->y2==s2->y2)
		return true;
	else
		return false;
}

bool UtilCpp::isInside(square *sqr, unsigned short x, unsigned short y)
{
	if(x>=sqr->x1 && x<=sqr->x2 && y>=sqr->y1 && y<=sqr->y2)
		return true;
	else
		return false;
}

vector<pixel> UtilCpp::bresenhamline(int x0, int y0, int x1, int y1, int max)
{
//	pixel *p;
	vector<pixel> p;

	 bool steep = abs(y1 - y0) > abs(x1 - x0);
	 if(steep)
	 {
		 SWAP(x0, y0)
		 SWAP(x1, y1)
	 }
	 if(x0 > x1)
	 {
		 SWAP(x0, x1)
		 SWAP(y0, y1)
	 }
	 int deltax = x1 - x0;
	 int deltay = abs(y1 - y0);
	 int error = deltax / 2;
	 int ystep;
	 int y = y0;
	 if (y0 < y1)
		 ystep = 1;
	 else
		 ystep = -1;
	 int x, index;

	 for(x=x0,index=0; x<=x1; x++,index++)
	 {
		 if(steep)
		 {
			 p[index].j=y;
			 p[index].i=x;
		 }
		 else
		 {
			 p[index].j=x;
			 p[index].i=y;
		 }
		 error = error - deltay;
		 if (error < 0)
		 {
			 y = y + ystep;
			 error = error + deltax;
		 }
	 }
	 return p;
}

#define DEBUG
vector<Point> UtilCpp::getOuterContour(Mat & src, int x0, int y0, int x1, int y1)
{
	vector<Point> vp;
#ifdef DEBUG
	Mat dbg = src.clone();
#endif
	cv::Point p, lastp, lastpr;
	uchar currs;
	bool foundBreak=false;

	if(src.channels()>1)
	{
		cerr << "src.channels()>1" << endl;
		return vp;
	}

	 bool steep = abs(y1 - y0) > abs(x1 - x0);
	 if(steep)
	 {
		 SWAP(x0, y0)
		 SWAP(x1, y1)
	 }
	 if(x0 > x1)
	 {
		 SWAP(x0, x1)
		 SWAP(y0, y1)
	 }
	 int deltax = x1 - x0;
	 int deltay = abs(y1 - y0);
	 int error = deltax / 2;
	 int ystep;
	 int y = y0;
	 if (y0 < y1)
		 ystep = 1;
	 else
		 ystep = -1;
	 int x, index;
	 if(x0>=0 && y0>=0 && x0<src.cols && y0<src.rows)
		 currs = src.at<uchar>(x0,y0);

	 for(x=x0,index=0; x<=x1; x++,index++)
	 {
		 lastp=p;

		 if(steep)
		 {
			 p.x=y;
			 p.y=x;
		 }
		 else
		 {
			 p.x=x;
			 p.y=y;
		 }
		 if(lastp.x>=0 && lastp.y>=0 && lastp.x<=src.cols &&  lastp.y<=src.rows)
			 lastpr=lastp;
		 if(p.x>=0 && p.y>=0)
		 {
			 if(p.x>=src.cols || p.y>=src.rows)
			 {
				 foundBreak=true;
				 break;
			 }
#ifdef DEBUG
			 dbg.at<uchar>(p)=200;
#endif
			 if(src.at<uchar>(p)==0)
			 {
				 if(currs==255)
					 vp.push_back(lastp);
				 currs=0;
			 }
			 else
				 currs=255;
		 }
		 error = error - deltay;
		 if (error < 0)
		 {
			 y = y + ystep;
			 error = error + deltax;
		 }
	 }
	 if(foundBreak && currs==255)
		 vp.push_back(lastpr);
//	 if(vp.size()==0)
//	 {
//		 cout << "vp.size=" << vp.size() << endl;
	 imshow("dbg",dbg);
	 waitKey(40);
//	 Mat dbg3c = convertTo3Channels(dbg);
//	 writeJPG(string("/home/alex/TestData/AVSS/ghost/border"),dbg3c,countFrames++);
//	 }
	 return vp;
}

int UtilCpp::countOnLine(Mat & src, int x0, int y0, int x1, int y1, int checkValue)
{
//	pixel *p;
	//vector<pixel> p;
	Point2i p;
	Mat dbg = src.clone();
	if(src.channels()>1)
	{
		cerr << "src.channels()>1" << endl;
		return -1;
	}
	 bool steep = abs(y1 - y0) > abs(x1 - x0);
	 if(steep)
	 {
		 SWAP(x0, y0)
		 SWAP(x1, y1)
	 }
	 if(x0 > x1)
	 {
		 SWAP(x0, x1)
		 SWAP(y0, y1)
	 }
	 int deltax = x1 - x0;
	 int deltay = abs(y1 - y0);
	 int error = deltax / 2;
	 int ystep;
	 int y = y0;
	 if (y0 < y1)
		 ystep = 1;
	 else
		 ystep = -1;
	 int x, index;
	 int count=0;
	 for(x=x0,index=0; x<=x1; x++,index++)
	 {
		 if(steep)
		 {
			 p.x=y;
			 p.y=x;
		 }
		 else
		 {
			 p.x=x;
			 p.y=y;
		 }
		 if(src.at<uchar>(p)==checkValue)
			 count++;
		 dbg.at<uchar>(p)=150;

		 error = error - deltay;
		 if (error < 0)
		 {
			 y = y + ystep;
			 error = error + deltax;
		 }
	 }
	 imshow("dbg",dbg);
	 waitKey(0);
	 return count;
}

Mat UtilCpp::sideBySideImgs(Size imgsize, Mat & m1, Mat & m2, Mat & m3, Mat & m4, bool resizing)
{
	cv::Size dsize(2*imgsize.width,2*imgsize.height);
	Mat combined(dsize, m1.type(), Scalar(0));
	Rect r1(0,0,imgsize.width,imgsize.height);
	Rect r2(0,imgsize.height, imgsize.width,imgsize.height);
	Rect r3(imgsize.width,0,m3.cols,m3.rows);
	Rect r4(imgsize.width,imgsize.height,m4.cols,m4.rows);
	Mat roi1(combined,r1);
	Mat roi2(combined,r2);
	Mat roi3(combined,r3);
	Mat roi4(combined,r4);
	Mat res1, res2;
	if(resizing)
	{
		resize(m1, res1, imgsize, 0, 0, INTER_LINEAR);
		resize(m2, res2, imgsize, 0, 0, INTER_LINEAR);
		res1.copyTo(roi1);
		res2.copyTo(roi2);
	}
	else
	{

	}
	m3.copyTo(roi3);
	m4.copyTo(roi4);
	return combined;
}

Mat UtilCpp::convertTo8b(Mat & src)
{
	double min, max;
	cv::minMaxIdx(src,&min,&max);
	Mat result(src.size(),CV_8U);
	if(max>=0)
		src.convertTo(result,CV_8U,255./max);
	else
		result = Mat::ones(src.size(),CV_8U);

	return result;
}

Mat UtilCpp::convertToPositive(Mat & src)
{
	double min, max;
	cv::minMaxIdx(src,&min,&max);
	int c, i, j;
	float a;
	Mat_<uchar> dst = Mat(src.size(), CV_8U);

	c=(int)-min;
	a=255./(max-min);
	if(a<0 || max>1000)
	{
		dst.setTo(0);
		return dst;
	}

	short d,s;
	for(i=0; i<src.rows; i++)
	{
		for(j=0; j<src.cols; j++)
		{
			s=src.at<char>(i,j);
			d=(s+c)*a;
//			if(d>0)
//				cout << d << endl;
			dst.at<uchar>(i,j) =d;
		}
	}
	return dst;
}


void UtilCpp::writeToLogFile(int n, bool endline)
{
	FILE * pFile;
	pFile = fopen ("log.txt","a");
	char buffer[2];
//	itoa(n,buffer,10);
	sprintf(buffer,"%d",n);
	if (pFile!=NULL)
	{
		fputs(buffer,pFile);
		fclose (pFile);
	}
}

Mat UtilCpp::getFrame(bool hsv, int currfile,  OOTestCase * tc)
{
	Mat img;
	if(tc->getTestset() == LIVE_CAM)
		img=tc->next();
	else
		img=getFrame(hsv, tc->getPath(), currfile, tc->getFnlength(), tc->getExt());

	return img;
}

Mat UtilCpp::getFrame(bool hsv, string path, int currfile, int fnlength, string ext)
{
	char filename[200];
	char temp_path[200];
	strcpy(temp_path, path.c_str());
	createFilename(&temp_path[0], &filename[0], currfile, fnlength, ext);
	Mat src = imread(&temp_path[0]);

	if( !src.data ) // check if the image has been loaded properly
	{
//		cerr << "error loading image." << &temp_path[0] << endl;
	}

	if(hsv)
	{
		Mat_<Vec3b> srchsv;
		cvtColor(src, srchsv, CV_BGR2HSV);
		src.release();
		return srchsv;
	}
	else
		return src;
}

void UtilCpp::drawText(Mat & src, int type, string text, float number, Point orig)
{

	int w=src.cols;
	int h=src.rows;
	if(w<80 || h<80)
		return;

	Scalar color;
	CvFont font;
	cvInitFont(&font,CV_FONT_HERSHEY_PLAIN|CV_FONT_ITALIC,1,1,0,1);
	int fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	int baseline=0;
	baseline += thickness;
	switch(type)
	{
		case TITLE:
			orig=Point(0.05*w,0.05*h);
			fontScale=1.4;
			if(src.channels()==3)
				color=Scalar(255,0,0);
			else
				color=Scalar(255);
			break;
		case SUB_TITLE:
			fontScale=1.2;
			orig=Point(0.05*w+2,0.05*h+30);
			if(src.channels()==3)
				color=Scalar(0,255,0);
			else
				color=Scalar(210);
			break;
		case SECTION:
			fontScale=1;
			orig=Point(0.05*w+2,0.05*h+55);
			if(src.channels()==3)
				color=Scalar(0,0,255);
			else
				color=Scalar(160);
			break;
		case SUB_SECTION:
			fontScale=0.8;
			orig=Point(0.05*w+2,0.05*h+75);
			if(src.channels()==3)
				color=Scalar(205,150,0);
			else
				color=Scalar(120);
			break;
	}

	if(number!=-1000000)
	{
		string number_in_text;
		stringstream ss; //create a stringstream
		ss << number; //add number to the stream
		number_in_text = ss.str();
		text.append(": ");
		text.append(number_in_text);
	}

	cv::putText(src, text, orig, fontFace, fontScale, Scalar(255,0,0), thickness, 8);
}
Mat UtilCpp::getRaimbowGradient(Mat hsv, Mat gray)
{

	//	#create the image arrays we require for the processing
	Mat hue,sat,val;
	Mat mask_1=Mat(hsv.size(),CV_8U);
	Mat mask_2=Mat(hsv.size(),CV_8U);

	//	#split image into component channels
	vector<Mat> planes;
	split(hsv, planes);
	hue=planes[0];
	sat=planes[1];
	val=planes[2];
	//	#rescale image_bw to degrees
	double max,min;
	cv::minMaxIdx(gray,&min,&max);
	gray*=360./max;
	//	cv::minMaxIdx(sat,&min,&max);

	//	#set the hue channel to the greyscale image
	//	cout << max << endl;
	gray.copyTo(hue);
	//	#set sat and val to maximum
	sat.setTo(1);
	val.setTo(1);

	//	#adjust the pseudo color scaling offset, 120 matches the image you displayed
	int offset=150;
	compare(hue,360-offset, mask_1, CV_CMP_GE);
	compare(hue,360-offset, mask_2, CV_CMP_LT);
	add(hue,offset-360,hue,mask_1);
	add(hue,offset,hue,mask_2);

	//	#merge the channels back
	int ch[] = {0,0,1,1,2,2};
	vector<Mat> vres;
	Mat mixedHsv = Mat(hsv.size(), hsv.type());
	vres.push_back(mixedHsv);
	mixChannels( planes, vres, ch, 3 );

	//	#convert back to RGB color space, for correct display
	Mat result;
	cvtColor(vres[0],result,CV_HSV2RGB);

	return result;
}

Mat UtilCpp::convertTo3Channels(Mat & src)
{
	if(src.channels()!=1)
		return Mat();

	vector<Mat> planes;
	planes.push_back(src);
	planes.push_back(src);
	planes.push_back(src);
	int ch[] = {0,0,1,1,2,2};
	vector<Mat> vres;
	Mat mixed = Mat(src.size(), CV_8UC3);
	vres.push_back(mixed);
	mixChannels( planes, vres, ch, 3 );
	return mixed;
}

Mat UtilCpp::getOneChHistogram( Mat src, float tc, float MAD, int type, vector<float> &hd, bool render )
{
	Mat dst;
	Mat ret, resizImg;

	/// Separate the image in 3 places ( B, G and R )
	Scalar mean, mean2;
	Scalar std, std2;
	float tcPlot, MADplot;

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist;
	switch(type)
	{
	case CH_HSV:
//		src=src*256./2;
		tcPlot=tc*256;
		MADplot=MAD*256;
		break;
	case CH_RGB:
	default:
		tcPlot=tc;
		break;
	}

	calcHist( &src, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 256; int hist_h = 256;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	/// Draw for each channel
	Vec4f fit;
	vector<Point2f> points=vector<Point2f>(histSize);
	for( int i = 1; i < histSize; i++ )
	{
		points[i]=Point2f(i,cvRound(b_hist.at<float>(i)));
		//	  cout << points[i] << endl;
		if(render)
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
				Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
				Scalar( 255, 0, 0), 1, 8, 0  );
	}

	Mat newHist;
	b_hist.copyTo(newHist);
	newHist.at<float>(0)=0;
	newHist.at<float>(1)=0;
	newHist.at<float>(2)=0;

	meanStdDev(newHist,mean,std);
	meanStdDev(b_hist,mean2,std2);

	uchar m,s,m2,s2;
	m=(uchar)mean[0];
	s=(uchar)std[0];

	m2=(uchar)mean2[0];
	s2=(uchar)std2[0];
	if(render)
	{
		line(histImage, Point(s2,0), Point(s2,255), Scalar(0,255,0), 1, 8, 0);
//		line(histImage, Point(s,0), Point(s,255), Scalar(255,255,0), 1, 8, 0);
//		line(histImage, Point(m,0), Point(m,255), Scalar(255,255,255), 1, 8, 0);
		line(histImage, Point(m2,0), Point(m2,255), Scalar(0,0,255), 1, 8, 0);
	}

	Mat fline= Mat(points,true);
	vector<Point2f> error = vector<Point2f>(points);
	int baseline=0; int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.7; int thickness = 1;
	baseline += thickness;
	Point textOrg(20,30);
	Point textOrg2(20,50);
	float errormse, mst;
	string mse, met; stringstream ss, ss2;//create a stringstream

	switch(type)
	{
	case CH_HSV:
//		fitLine(fline,fit,CV_DIST_L2,0,0.01,0.01);
//		float a,b;
//		uchar extr;
//		a=fit[1]/fit[0];
//		b=fit[3]-a*fit[2];
//		extr=(uchar)(a*255+b);
//		if(render)
//			line(histImage, Point(0,255-(uchar)b), Point(255,255-extr), Scalar(0,0,255), 1, 8, 0);
//		for( int i = 1; i < histSize; i++ )
//		{
//			float predict=getLinePoint(a,b,i);
//			error[i]=Point2f(points[i].y,predict);
//		}
//		errormse=calcMSE(error);
//		mst=errormse/a;
//		ss << errormse;//add number to the stream
//		ss2 << mst;
//		mse = ss.str();//return a string with the contents of the stream
//		met = ss2.str();
		hd[0]=std[0];
		hd[1]=std2[0];
//		hd[2]=errormse;
//		hd[3]=mst;
//		if(render)
//		{
//			putText(histImage, mse, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
//			putText(histImage, met, textOrg2, fontFace, fontScale, Scalar::all(255), thickness, 8);
//		}
		ret=histImage;
		break;
	case CH_RGB:
	default:
//		line(histImage, Point(0,m2), Point(255,m2), Scalar(255,0,255), 1, 8, 0);
//		line(histImage, Point(0,m), Point(255,m), Scalar(255,255,255), 1, 8, 0);
		resizImg=histImage.colRange(0,150);
		ret=Mat(histImage.size(),histImage.type());
		resize(resizImg,ret,ret.size());
		break;
	}
	if(render)
	{
		line(histImage, Point(MADplot,0), Point(MADplot,255), Scalar(255,255,0), 1, 8, 0);
		line(histImage, Point(tcPlot,0), Point(tcPlot,255), Scalar(255,0,0), 1, 8, 0);
	}
	return ret;
}

Mat UtilCpp::getRGBHistogram( Mat src, vector<float> &hd, bool render )
{
	Mat dst;

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	Scalar meanr, meang, meanb;
	Scalar stdr, stdg, stdb;


	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 256; int hist_h = 256;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	meanStdDev(r_hist,meanr,stdr);
	meanStdDev(r_hist,meang,stdg);
	meanStdDev(r_hist,meanb,stdb);

	hd[0]=meanr[0];
	hd[1]=stdr[0];

	/// Draw for each channel
	if(render)
	{
		for( int i = 1; i < histSize; i++ )
		{
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
					Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
					Scalar( 255, 0, 0), 1, 8, 0  );
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
					Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
					Scalar( 0, 255, 0), 1, 8, 0  );
			line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
					Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
					Scalar( 0, 0, 255), 1, 8, 0  );
		}

		uchar mr,mg,mb,sr,sg,sb;
		mr=255-(uchar)meanr[0];	mg=255-(uchar)meang[0];	mb=255-(uchar)meanb[0];
		sr=(uchar)stdr[0];	sg=(uchar)stdg[0];	sb=(uchar)stdb[0];

		line(histImage, Point(0,mr), Point(255,mr), Scalar(255,255,255), 1, 8, 0);
		line(histImage, Point(0,mg), Point(255,mg), Scalar(255,255,255), 1, 8, 0);
		line(histImage, Point(0,mb), Point(255,mb), Scalar(255,255,255), 1, 8, 0);

		line(histImage, Point(sr,0), Point(sr,255), Scalar(255,255,0), 1, 8, 0);
		line(histImage, Point(sg,0), Point(sg,255), Scalar(255,255,0), 1, 8, 0);
		line(histImage, Point(sb,0), Point(sb,255), Scalar(255,255,0), 1, 8, 0);
	}
	return histImage;
}

int UtilCpp::getLinePoint(float a, float b, int x)
{
	return a*x+b;
}

float UtilCpp::calcMSE(vector<Point2f> v)
{
	float SE=0.0;
	for(uint i=0;i<v.size();i++)
	{
		SE+=pow(v[i].y-v[i].x,2);
	}
	return SE/v.size();
}

void UtilCpp::removeNearbyNoise(Mat & src, bool preserveOrig, int cmin, int sumth)
{
	Mat tmp;
	src.copyTo(tmp);

	vector<Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	Mat newSegImage = Mat::zeros(src.size(),src.type());
	try {
		findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	} catch (cv::Exception & e) {
		cout << e.what() << endl;
		return;
	}
	if(contours.size()<=0)
		return;

	vector<vector<Point> >::const_iterator itc = contours.begin();

	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0], itc++)
	{
		int n = (int)contours[idx].size();
		if (n > cmin)
		{
			Rect r0= boundingRect(Mat(*itc));
			Mat seg;
			Mat roi = Mat(newSegImage, r0);
			Mat closedImg, E;
			seg = Mat(src,r0);
			erode(seg,E,Mat(),Point(-1,-1),2);
			int sumeroded;
			sumeroded = sum(E)[0]/255;
			if(sumeroded<sumth)
			{
				continue;
			}
			const Point* p = &contours[idx][0];
			if(p==NULL)
				continue;
			fillPoly(newSegImage, &p, &n, 1, Scalar(255));
			if(preserveOrig)
				newSegImage = newSegImage & src;
		}
	}
	newSegImage.copyTo(src);
}

Mat UtilCpp::removeNoise(Mat src)
{
	Mat srccp;
	src.copyTo(srccp);
	vector<vector<Point> > contours;
	findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	unsigned int cmin= 15; // minimum contour length
	vector<vector<Point> >::const_iterator itc = contours.begin();
	Mat newSegImage = Mat::zeros(src.size(),src.type());
	int i=-1;
	while (itc!=contours.end())
	{
		i++;
		if (itc->size() > cmin)
		{
			Rect r0= boundingRect(Mat(*itc));
			Mat seg;
			//Mat roi = Mat(newSegImage, r0);
			Mat closedImg, E;
			seg = Mat(srccp,r0);
			erode(seg,E,Mat(),Point(-1,-1),1);
			int sumeroded;
			sumeroded = sum(E)[0]/255;
			if(sumeroded<4)
			{
				itc++;
				continue;
			}
			drawContours( newSegImage, contours, i, Scalar(255), CV_FILLED);
			//seg.copyTo(roi);
			itc++;
			continue;
		}
		itc++;
	}
	return newSegImage;
}

void UtilCpp::zeroNegatives(vector<float> & v)
{
	for(uint i=0; i<v.size(); i++)
	{
		if(v[i]<0)
			v[i]=0;
	}
}
void UtilCpp::capHoles(Mat & src)
{
	Mat tmp;
	src.copyTo(tmp);

	vector<Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	try {
		findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	} catch (cv::Exception & e) {
		cout << e.what() << endl;
		return;
	}
	if(contours.size()<=0)
		return;
	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
	{
		if(hierarchy[idx][2]>0)
		{
			const Point* p = &contours[hierarchy[idx][2]][0];
			if(p==NULL)
				continue;
			int n = (int)contours[hierarchy[idx][2]].size();
			fillPoly(src, &p, &n, 1, Scalar(255));
			int brotherIdx=hierarchy[idx][2];
			while(hierarchy[brotherIdx][0]>0)
			{
				const Point* p = &contours[hierarchy[brotherIdx][0]][0];
				if(p==NULL)
				{
					brotherIdx=hierarchy[brotherIdx][0];
					continue;
				}
				int n = (int)contours[hierarchy[brotherIdx][0]].size();
				fillPoly(src, &p, &n, 1, Scalar(255));
				brotherIdx=hierarchy[brotherIdx][0];
			}
		}
	}
}

void UtilCpp::applySobel(Mat & src)
{
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;

	  /// Gradient X
	  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	  Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  /// Gradient Y
	  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  /// Total Gradient (approximate)
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src );
//	  magnitude(grad_x,grad_y,src);
}


void UtilCpp::applyScharr(Mat & src)
{
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;

	  /// Gradient X
	  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	  Scharr( src, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  /// Gradient Y
	  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  Scharr( src, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  /// Total Gradient (approximate)
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src );
}

Mat UtilCpp::highPass(Mat & src)
{
	Mat result(src.size(),src.type());
	Mat kernel = (Mat_<char>(3,3)<< 0, -13,  0,
	                               -13,  53, -13,
	                                0, -13,  0);

	src.copyTo(result);
	filter2D(src, result, src.depth(), kernel, Point(0,0));
	return result;
}

void UtilCpp::save(float * src, int n, string & path)
{
	  ofstream outfile(path.c_str(),ofstream::binary);
	  for(int i=0; i<n; i++)
	  {
		  outfile << *src++;
	  }
	  outfile.close();
}

float * UtilCpp::load(int n, string & path)
{
	float * result = new float[n];
	ifstream infile (path.c_str(),ifstream::binary);
	for(int i=0; i<n; i++)
	{
	  infile >> result[i];
	}
	infile.close();
	return result;
}

bool UtilCpp::compareArray(float *src1, float *src2, int n)
{
	bool equal=true;
	float * result = new float[n];
	for(int i=0; i<n; i++)
	{
		if(*src1++!=*src2++)
		{
			equal=false;
			cout << "different array at " << i << endl;
		}
	}
	return equal;
}

bool UtilCpp::rectAtBorder(Rect& r, Rect & world, int minDiagDistance) {
	bool atborder = (r.x <= minDiagDistance || r.y <= minDiagDistance
			|| (r.x + r.width) >= world.width - minDiagDistance
			|| (r.y + r.height) >= world.height - minDiagDistance);
	return atborder;
}

Mat UtilCpp::getLargestContour(Mat & mask, Rect & largestRect, vector<Point> & lvp, int * count)
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

string UtilCpp::floatToString(vector<float> vf, int precision)
{
	string result;
	for(uint i=0; i<vf.size(); i++)
	{
		stringstream ss (stringstream::in | stringstream::out);
		if(vf[i]>1000)
			ss << std::fixed << std::setprecision(0) << vf[i];
		else
			ss << std::setprecision(precision) << vf[i];
		result = result + "," + ss.str();
	}
	return result;
}

string UtilCpp::floatToString(float f)
{
	string result;
	stringstream ss (stringstream::in | stringstream::out);
	ss << f;
	result = ss.str();
	return result;
}

string UtilCpp::Vec3fToString(Vec3f f)
{
	string result;
	stringstream ss (stringstream::in | stringstream::out);
	ss << f.val[0] << "," << f.val[1] << "," << f.val[2];
	result = ss.str();
	return result;
}

string UtilCpp::intToString(int n)
{
	string result;
	stringstream ss (stringstream::in | stringstream::out);
	ss << n;
	result = ss.str();
	return result;
}

string UtilCpp::boolToString(bool b)
{
	int n;
	n=b ? 1 : 0;
	string result;
	stringstream ss (stringstream::in | stringstream::out);
	ss << n;
	result = ss.str();
	return result;
}

void UtilCpp::vectorToCSV(string file,vector<float> v)
{
	ofstream CSVToFile(file.c_str(), ios::out | ios::binary);
	    //////////////////vector element to CSV////////////////////////////////////////
	for (std::vector<float>::iterator i = v.begin(); i != v.end(); i++)
	{
	    CSVToFile << *i;
	    CSVToFile << "\n";
	}
	CSVToFile.close();
}
void UtilCpp::printVector( vector<int> & v)
{
	printf("\n");
	for(uint i=0; i<v.size(); i++)
	{
		printf("%d ", v[i]);
	}
	fflush(stdout);
}

void UtilCpp::printVector( vector<float> & v, bool withcomma)
{
	if(withcomma)
	{
		for(uint i=0; i<v.size()-1; i++)
		{
			//printf("%f ", v[i]);
			cout << v[i] << ",";
		}
		cout << v[v.size()-1];
	}
	else
	{
		for(uint i=0; i<v.size(); i++)
		{
			//printf("%f ", v[i]);
			cout << v[i];
		}
	}
	printf("\n");
	fflush(stdout);
}

void UtilCpp::printVector( vector<string> & v)
{
	printf("\n");
	for(uint i=0; i<v.size()-1; i++)
	{
		cout << v[i] << ",";
	}
	if(v.size()>0)
		cout << v[v.size()-1] << endl;
	fflush(stdout);
}

void UtilCpp::printArray( float * v, int n)
{
	printf("\n");
	for(int i=0; i<n; i++)
	{
		printf("%f ", v[i]);
	}
	fflush(stdout);
}
void UtilCpp::appendToFile(map<string,string> vs, string path)
{
	std::ofstream file(path.c_str(), std::ios_base::app | std::ios_base::out);
	file << std::fixed;
	file.precision(4);
	for(map<string,string>::iterator iter = vs.begin(); iter != vs.end(); ++iter)
	{

		file << iter->first << "," << iter->second;
	}
	file.flush();
	file.close();
}

void UtilCpp::appendToFile(vector<string> & vs, string path, bool addComma, uint ncache)
{
	if(vs.size()>=ncache)
	{
		std::ofstream file(path.c_str(), std::ios_base::app | std::ios_base::out);
		file << std::fixed;
		file.precision(4);
		if(addComma)
		{
			for(uint i=0; i<vs.size(); i++)
			{
				file << vs[i] << ",";
			}
		}
		else
		{
			for(uint i=0; i<vs.size(); i++)
			{
				file << vs[i];
			}
		}
		file.flush();
		file.close();
		//vs.clear();
	}
}

void UtilCpp::appendToFile(string s, string path)
{
	std::ofstream file(path.c_str(), std::ios_base::app | std::ios_base::out);
	file.precision(4);
	file << std::fixed;
	file << s;
	file.flush();
	file.close();
}

Mat UtilCpp::bgr2hsv(Mat input)
{
	Mat conv, hsv32f;
	input.convertTo(conv,CV_32F);
	conv *= 1./255;
	cvtColor(conv, hsv32f, CV_BGR2HSV);
	return hsv32f;
}

Mat UtilCpp::hsv2bgr(Mat input)
{
	Mat conv, rgb;
	input = input * 255;
	input.convertTo(conv,CV_8U);
	cvtColor(conv, rgb, CV_HSV2BGR);
	return rgb;
}

int UtilCpp::histo_bin( float h, float s, float v )
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
/*
void UtilCpp::Get_Pixel_Bin ( Mat & I , Mat & bin_mat, int nbins)
{
	float bin_width = floor ( 256. / (float)(nbins) );

	for (int row=0; row<bin_mat.rows; row++)
	for (int col=0; col<bin_mat.cols; col++)
	{
		Vec3f bv = I.at<Vec3f>( row , col );
		int b = histo_bin(bv.val[0],bv.val[1],bv.val[2]);
		bin_mat.at<int>(row,col)=b;
	}
	return;
}
vector<Mat> UtilCpp::getIH(Mat & I, int nbins)
{
	vector<Mat> IIH;
	for (uint i=0; i<IIH.size(); i++) {
		IIH[i].release();
	}
	if(IIH.size()>0)
		IIH.clear();

	for (int i=0; i<nbins; i++) {
		Mat curr_ii = Mat(I.rows,I.cols,CV_32S);
		IIH.push_back(curr_ii);
	}
	compute_IH ( I , &IIH, nbins);
	return IIH;
}

// compute_IH - compute integral histogram. Also possible to use OpenCV's
// routine, however there is a difference in the size of matrices returned
bool UtilCpp::compute_IH( Mat & I , vector < Mat >* vec_II, int nbins )
{
	//int save_dbg = dbg;

	//reset IIV matrices
	vector < Mat >::iterator it;
	for ( it = vec_II->begin() ; it != vec_II->end() ; it++ ) {
		it->setTo(0);
	}

	Mat curr_bin_mat;
	curr_bin_mat = Mat(I.rows,I.cols,CV_32S);
	Get_Pixel_Bin ( I , curr_bin_mat, nbins);
	//fill matrices
	int i , j , currBin, count;
	double vup , vleft , vdiag , z;
	for ( i = 0 ; i < I.rows ; i++ ) {
		for ( j = 0 ; j < I.cols ; j++ ) {
			currBin = curr_bin_mat.at<int>(i,j);

			for ( it = vec_II->begin() , count = 0 ; it != vec_II->end() ; it++ , count++ ) {

				if ( i == 0 ) {//no up
					vup = 0;
					vdiag = 0;
				} else {
					vup = it->at<int>( i-1 , j );
				}
				if ( j == 0 ) {//no left
					vleft = 0;
					vdiag = 0;
				} else {
					vleft = it->at<int>( i , j-1 );
				}
				if ( i > 0 && j > 0 ) {//diag exists
					vdiag = it->at<int>( i-1 , j-1 );
				}
				//set cell value
				z = vleft + vup - vdiag;
				if ( currBin == count )
					z++;
				it->at<int>( i , j ) = (int)z;
			}//next it
		}//next j
	}//next i

	curr_bin_mat.release();
	return true;
}
*/
bool UtilCpp::compute_histogram ( int tl_y , int tl_x , int br_y , int br_x , vector < Mat >* iiv , vector < float >& hist)
{
	vector < Mat >::iterator it;
	hist.clear();
	float left , up , diag;
	double sum = 0;
	float z;
	int c=0;

	for ( it = iiv->begin() ; it != iiv->end() ; it++ ) {
		c++;
		if ( tl_x == 0 ) {
			left = 0;
			diag = 0;
		} else {
			left = it->at<int>( br_y , tl_x - 1 );
		}

		if ( tl_y == 0 ) {
			up = 0;
			diag = 0;
		} else {
			up = it->at<int>( tl_y - 1 , br_x );
		}
		if ( tl_x > 0 && tl_y > 0 ) {
			diag = it->at<int>( tl_y - 1 , tl_x - 1 );
		}
		z = (float)it->at<int>( br_y , br_x ) - left - up + diag;
		hist.push_back(z);
		sum += z;
	}

	vector < float >::iterator it2;
	for ( it2 = hist.begin() ; it2 != hist.end() ; it2++ ) {
		(*it2) /= sum;
	}
	return true;
}

Vec3f UtilCpp::fitLine(double *x, double *y, int n)
{
	double c0, c1, cov00, cov01, cov11, sumsq;
	size_t xstride = 1, ystride = 1;
	gsl_ieee_env_setup();
	gsl_fit_linear(x, xstride, y, ystride, n,  &c0, &c1, &cov00, &cov01, &cov11, &sumsq);
	Vec3f result(c0,c1,sumsq);
	return result;
}

void UtilCpp::plotLineAndPoints(Mat & src, Vec3f line, vector<double> & x, vector<double> & y)
{
	int n = x.size();
	Point p1(x[0],line[0]+line[1]*x[0]);
	Point p2(x[n-1],line[0]+line[1]*x[n-1]);
	cv::line(src,p1,p2,Scalar(255,0,0));
	for(int i=0; i<n; i++)
	{
		Point p(x[i],line[0]+line[1]*x[i]);
		circle(src,p,0.2,Scalar(0,255,0));
	}
}

bool UtilCpp::fileExists(string name) {
	ifstream infile(name.c_str());
	return infile.good();
}

Mat UtilCpp::vecToMat(vector<float> v)
{
	Mat ret(1,v.size(),CV_32F);
	for(uint i=0; i<v.size(); i++)
	{
		ret.at<float>(0,i)=v[i];
	}
	return ret;
}


float UtilCpp::weightedMedian(vector<float> x, vector<float> w) {
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

float UtilCpp::median(vector<float> vec) {
	typedef vector<float>::size_type vec_sz;
	vec_sz size = vec.size();
	if (size == 0)
	{
		//throw domain_error("median of an empty vector");
		//cerr << "median of an empty vector" << endl;
		return 0;
	}

	sort(vec.begin(), vec.end());
	vec_sz mid = size / 2;
	return size % 2 == 0 ? (vec[mid] + vec[mid - 1]) / 2 : vec[mid];
}

ushort UtilCpp::median(vector<ushort> vec) {
	typedef vector<ushort>::size_type vec_sz;
	vec_sz size = vec.size();
	if (size == 0)
	{
		//throw domain_error("median of an empty vector");
		//cerr << "median of an empty vector" << endl;
		return 0;
	}

	sort(vec.begin(), vec.end());
	vec_sz mid = size / 2;
	return size % 2 == 0 ? (vec[mid] + vec[mid - 1]) / 2 : vec[mid];
}

uchar UtilCpp::median(vector<uchar> vec) {
	typedef vector<uchar>::size_type vec_sz;
	vec_sz size = vec.size();
	if (size == 0)
	{
		//throw domain_error("median of an empty vector");
		//cerr << "median of an empty vector" << endl;
		return 0;
	}

	sort(vec.begin(), vec.end());
	vec_sz mid = size / 2;
	return size % 2 == 0 ? (vec[mid] + vec[mid - 1]) / 2 : vec[mid];
}

short UtilCpp::median(vector<short> vec) {
	typedef vector<short>::size_type vec_sz;
	vec_sz size = vec.size();
	if (size == 0)
	{
		//throw domain_error("median of an empty vector");
		//cerr << "median of an empty vector" << endl;
		return 0;
	}

	sort(vec.begin(), vec.end());
	vec_sz mid = size / 2;
	return size % 2 == 0 ? (vec[mid] + vec[mid - 1]) / 2 : vec[mid];
}
void UtilCpp::eraseByValue(std::vector<short> & myNumbers_in, int number_in)
{
    std::vector<short>::iterator iter = myNumbers_in.begin();
    while (iter != myNumbers_in.end())
    {
        if (*iter == number_in)
        {
            iter = myNumbers_in.erase(iter);
        }
        else
        {
           ++iter;
        }
    }
}

vector<uchar> UtilCpp::getMedian3ch(vector<vector<short> > v)
{
	vector<uchar> r(3);
	for(int i=0; i<3; i++)
	{
		eraseByValue(v[i],-1);
		r[i]=(uchar)median(v[i]);
	}
	return r;
}

float UtilCpp::medianOrLower(vector<float> vec) {
	typedef vector<float>::size_type vec_sz;
	vec_sz size = vec.size();
	if (size == 0)
	{
		//throw domain_error("median of an empty vector");
		cerr << "median of an empty vector" << endl;
		return -1;
	}

	sort(vec.begin(), vec.end());
	int thmin =  vec.size()/4;
	vec.erase(vec.end()-thmin,vec.end());
	vec_sz mid = size / 2;
	return size % 2 == 0 ?  vec[mid + 1] : vec[mid];
}

float UtilCpp::getMean(vector<float> v)
{
	float sum = std::accumulate(v.begin(), v.end(), 0.0);
	float mean = sum / v.size();
	return mean;
}

int UtilCpp::getMean(vector<ushort> v)
{
	int sum = std::accumulate(v.begin(), v.end(), 0.0);
	int mean = sum / v.size();
	return mean;
}


float UtilCpp::stdDev(vector<float> v)
{
	double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
	double m =  sum / v.size();

	double accum = 0.0;
	std::for_each (std::begin(v), std::end(v), [&](const double d) {
	    accum += (d - m) * (d - m);
	});

	int n=MAX(1,v.size()-1);
	float stdev = sqrt(accum /n);
	return stdev;
}

Mat UtilCpp::equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

Mat UtilCpp::getLRG(Mat & src)
{
	Mat_<Vec3b> & img = (Mat_<Vec3b>&)src;
	//Mat_<Vec3b>& vroiImg = (Mat_<Vec3b>&)roiImg;
	Mat_<Vec3b> lrg(src.size(),CV_8UC3);
	float teta = 43.58*CV_PI/180.0;
	float cost = std::cos(teta);
	float sent = std::sin(teta);// b g r
	for(int i=0; i<src.rows; i++)
	{
		for(int j=0; j<src.cols; j++)
		{
			Vec3b p = img.at<Vec3b>(i,j);
			uchar l;
//			if(p.val[1]==0)
//				l = cost*std::log(p.val[2]/1E-10) + sent*std::log(p.val[0]/1E-10);
//			else
//				l = cost*std::log(p.val[2]/p.val[1]) + sent*std::log(p.val[0]/p.val[1]);

			uchar L=0.2116*p.val[2]+0.7152*p.val[1]+0.0722*p.val[0];
			float sqrtcolor=sqrt(pow(p.val[2],2)+pow(p.val[1],2)+pow(p.val[0],2));
			uchar r=p.val[2]/(sqrtcolor);
			uchar g=p.val[1]/(sqrtcolor);
			//uchar b=p.val[0]/(sqrtcolor);
			Vec3b d(L,g,r);
			lrg.at<Vec3b>(i,j)=d;
		}
	}
	return lrg;
}
/*
Mat UtilCpp::lsbpEdges(Mat & src, int th, int type)
{
	ULBSobel ulb;
	Mat edges = Mat::zeros(src.size(),CV_32S);
	int bordersize;
	if(type==0)
		bordersize=3;
	else
		bordersize=4;
	vector<KeyPoint> kps = ulb.mat2Keypoints(src,th);
	KeyPointsFilter::runByImageBorder(kps,src.size(),bordersize);
	vector<KeyPoint>::iterator it;
	int recl=2*bordersize+1;
	for( it= kps.begin(); it!= kps.end();it++)
	{
		Rect rroi((int)it->pt.x-bordersize,(int)it->pt.y-bordersize,recl,recl);
		Mat roi(src,rroi);
		int gx = ulb.xEdgeMag(roi,th);
		int gy = ulb.yEdgeMag(roi,th);
	    //cout << "(gx,gy)=" << gx << "," << gy << endl;
	    int norm=(int)sqrt(gx*gx+gy*gy);
	    edges.at<int>((int)it->pt.y,(int)it->pt.x)=norm;
	}
	Mat edges8b = convertTo8b(edges);
	return edges8b;
}
*/

vector<vector<ushort> > UtilCpp::getLbspSet(vector<KeyPoint> vp, vector<KeyPoint> vref, Mat & src)
{
	vector<vector<ushort> > fset;

	for (uint i=0; i<vref.size(); i++) {
		int x = vref[i].pt.x;//+roiback.x;
		int y = vref[i].pt.y;//+roiback.y;
		const size_t idx_uchar = src.cols*y + x;
		const size_t idx_uchar_rgb = idx_uchar*3;
		const uchar* u = src.data+idx_uchar_rgb;
		vector<vector<ushort> > f = getLBSP(vp, src, u);
		fset.insert(fset.end(),f.begin(),f.end());
	}
	return fset;
}


vector<vector<ushort> > UtilCpp::getLBSP(vector<KeyPoint> vp, Mat & src, const uchar * ref)
{
	int th=50;
	//const uchar ur = (uchar)ref;
	vector<vector<ushort> > f;
	//KeyPointsFilter::runByImageBorder(vp,src.size(),2);

	for (uint i=0; i<vp.size(); i++) {
		int x = vp[i].pt.x;
		int y = vp[i].pt.y;
		const size_t idx_uchar = src.cols*y + x;
		const size_t idx_uchar_rgb = idx_uchar*3;
		const uchar* u = src.data+idx_uchar_rgb;

		vector<ushort> res(4);
		if(ref==NULL)
			ULBSP::computeRGBDescriptor(src,u,x,y,th,&res[0]);
		else
			ULBSP::computeRGBDescriptor(src,ref,x,y,th,&res[0]);

//		if(res[0]>35000 && res[1]>35000 && res[2]>35000)
//			cout << res[0] << "," << res[1] << "," << res[2] << endl;
		res[3]=abs(*u-*ref)+abs(*(u+1)-*(ref+1))+abs(*(u+2)-*(ref+2));
		f.push_back(res);
	}
	return f;
}

vector<KeyPoint> UtilCpp::getRandomKeyPoints(int w, int h, int grid)
	{
		int th=5;
		vector<KeyPoint> vp;
		int i;
		gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);
		int n=w*h/(grid*grid);//min(MIN(w,h),w*h/(grid*grid));
		for (i = 0; i < n; i++)
		{
			double v[2];
			gsl_qrng_get (q, v);
			cv::KeyPoint p(v[0]*w, v[1]*h,th);
			vp.push_back(p);
		}
		gsl_qrng_free (q);
		return vp;
	}

vector<Point> UtilCpp::getRandomPoints(int w, int h, int grid)
{
	vector<Point> vp;
	int i;
	gsl_qrng * q = gsl_qrng_alloc (gsl_qrng_sobol, 2);
	int n=MIN(30,w*h/(grid*grid));
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

void UtilCpp::randomPatches(int height, int width, float n, vector<Point> contour, vector<Rect>& patch_vec) {

#ifdef DEBUG
	Mat test=Mat::zeros(height,width,CV_8U);
	Mat points=Mat::zeros(height,width,CV_8U);
	Mat noFilter=Mat::zeros(height,width,CV_8U);
#endif
	Rect tworld(0,0,width,height);
	int slength=MIN(height,width)/(n*2);
	int slength2=MIN(height,width)/(n);
	Mat acum=Mat::zeros(height,width,CV_8U);

//	for(uint i=0; i<patch_vec.size(); i++)
//	{
//		Rect r(patch_vec[i].x-slength,patch_vec[i].y-slength,slength2,slength2);
//		r=r & tworld;
//		Mat roi(acum,r);
//		rectangle(acum,r,Scalar(1),1);
//	}
	for(uint i=0; i<contour.size(); i++)
	{
		Rect r(contour[i].x-slength,contour[i].y-slength,slength2,slength2);
		r=r & tworld;
		Mat roi(acum,r);
		int totalb = countNonZero(roi);
		if(totalb<r.width*r.height/20)
		{
			rectangle(acum,r,Scalar(1),1);
			patch_vec.push_back(r);
			//circle(points,contour[pos],0.4,Scalar(255),CV_FILLED);
		}
	}
#ifdef DEBUG
//	vector<vector<Point> > vvp;
//	vvp.push_back(contour);
//	drawContours(points,vvp,0,Scalar(150));
//	imshow("randomPatches",test);
//	imshow("sobol",points);
//	waitKey(0);
//	imshow("pmask",mask);
//	imshow("noFilter",noFilter);
#endif
}

#define DEBUG
int UtilCpp::evalOutContour(Mat & src, Mat & out, int angle_step, Point p, vector<Point> & selectedp)
{
	int R= sqrt(src.cols*src.cols+src.rows*src.rows);//min(src.cols/2,src.rows/2);
//#ifdef DEBUG
//		Mat dbg = Mat::zeros(src.cols,src.rows,CV_8U);
//#endif
	float pi=3.1415;
	float step = 2*pi/angle_step;
	float cx=p.x, cy=p.y;
	float x,y;
	int countNonZero=0;
#ifdef DEBUG
//	out.at<Vec3b>(p)=Vec3b(0,255,0);//255;
	circle(out,p,1,Scalar(0,255,0),CV_FILLED);
#endif
	for(int i=0; i<angle_step; i++)
	{
		float nstep = step*i;
		x = cx+R * cos(nstep);
		y = cy+R * sin(nstep);
		vector<Point> vp = getOuterContour(src,cx,cy,x,y);
		if(vp.size()>0)
			countNonZero++;
		for(uint j=0; j<vp.size(); j++)
		{
			if(vp[j].x<0 || vp[j].y<0 || vp[j].x>=out.cols || vp[j].y>=out.rows)
			{
//					cout << vp[j].x << "," << vp[j].y << endl;
				continue;
			}
#ifdef DEBUG
			//out.at<Vec3b>(vp[j])=Vec3b(255,0,0);//150;
			circle(out,vp[j],1,Scalar(255,0,0),CV_FILLED);
#endif
			selectedp.push_back(vp[j]);
		}

		//			else
		//				cout << "vp.size=" << vp.size() << endl;

	}

	return countNonZero;
}

float UtilCpp::evalOutContourSamples(Mat & src, Mat & ero, vector<Point> & vp, vector<Point> & selectedp, int angle_step, Mat out)
{
	int countContour=0;
	int countVp=0;
	for(uint i=0; i<vp.size(); i++)
	{
//		if(ero.at<uchar>(vp[i])==255)
//		{
			countContour+=evalOutContour(src,out,angle_step,vp[i], selectedp);
			countVp++;
#ifdef DEBUG
	Mat gout;
	resize(out, gout, Size(out.cols*2,out.rows*2), 0, 0, INTER_LINEAR);
	imshow("out",gout);
	waitKey(0);
#endif
//		}
	}
	float rate;
	if(countVp>0)
		rate=countContour/(float)(angle_step*countVp);
	else
		rate=2;



	return rate;
}

vector<Point> UtilCpp::sampleBorder(Mat borderMask, Mat ero)
{
	vector<Point> vp = getRandomPoints(borderMask.cols,borderMask.rows,5);
	vector<Point> selectedp;
	Mat outContour = Mat::zeros(borderMask.rows,borderMask.cols,CV_8U);
	evalOutContourSamples(borderMask,ero,vp,selectedp,50,outContour);
	return selectedp;
}

void UtilCpp::fixedPatches(int height, int width, Size grid, vector<Rect>& patch_vec) {
	if (!patch_vec.empty())
		patch_vec.clear();

	int w = floor(width / grid.width);
	int h = floor(height / grid.height);
	int y1, x1;
	int currw, currh;
	for (int i = 0; i < grid.height; i++) {
		y1 = h * i;
		if (i == grid.height - 1)
			currh = height - y1;
		else
			currh = h;

		for (int j = 0; j < grid.width; j++) {
			x1 = w * j;
			if (j == grid.width - 1)
				currw = width - x1;
			else
				currw = w;

			Rect p(x1, y1, currw, currh);
			patch_vec.push_back(p);
		}
	}
}
void UtilCpp::fixedPatches(int height, int width, int n, vector<Rect>& patch_vec, int type) {
	if (!patch_vec.empty())
		patch_vec.clear();

	int w = floor(width / n);
	int h = floor(height / n);
	int y1, x1;
	int x2, y2;
	int currw, currh;
	if (type == SQUARE_PATCHES) {
		for (int i = 0; i < n; i++) {
			y1 = h * i;
			if (i == n - 1)
				currh = height - y1;
			else
				currh = h;

			for (int j = 0; j < n; j++) {
				x1 = w * j;
				if (j == n - 1)
					currw = width - x1;
				else
					currw = w;

				Rect p(x1, y1, currw, currh);
				patch_vec.push_back(p);
			}
		}

	} else if (type == RECT_PATCHES) {
		Rect p;
		int ndx = (int) (((floor(((double) (((width)))) / 2.0))));
		int k;
		x2 = ndx;
		for (k = 1, x1 = 0, y1 = 0; k <= n; k++) {
			y2 = floor(k * height / n);
			Rect p1(0, y1, x2 - x1, y2 - y1);
			Rect p2(ndx, y1, x2 - x1, y2 - y1);
			patch_vec.push_back(p1);
			patch_vec.push_back(p2);
			y1 = y2;
		}
		// vertical patches
		int ndy = (int) (((floor(((double) (((height)))) / 2.0))));
		for (k = 1, x1 = 0, y1 = 0; k <= n; k++) {
			x2 = floor(k * width / n);
			y2 = ndy;
			Rect p1(x1, 0, x2 - x1, y2 - y1);
			Rect p2(x1, ndy, x2 - x1, y2 - y1);
			patch_vec.push_back(p1);
			patch_vec.push_back(p2);
			x1 = x2;
		}
	}
}

vector < Rect > UtilCpp::findBestPatches(vector<Size> psize, Mat& r1, Mat& r2, float disrate, vector<float> weights)
{
	float rate;
	RNG rng;

	vector<int> countValid(psize.size());
	vector< vector < Rect > > vecpatches(psize.size());
	vector < Rect > validpcs;
	for(uint ps=0; ps<psize.size(); ps++)
	{
		vector < Rect > pcs;
		fixedPatches(r1.rows, r1.cols, psize[ps], pcs);
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
	if(validpcs.size()==0)
		validpcs.push_back(Rect(0,0,r1.cols,r1.rows));
	return validpcs;
}

void UtilCpp::drawPatches(Mat & src , vector<Rect> & patches, Scalar color)
{
	for(uint i=0; i<patches.size(); i++)
	{
		rectangle(src,patches[i],color,1);
	}
}

uint UtilCpp::getHammingDiff(vector<ushort> v1, vector<ushort> v2)
{
	uint diff = (uint)hdist_ushort_8bitLUT(&v1[0],&v2[0]);
	return diff;
}

vector<ushort> UtilCpp::findMinDiff2(vector<vector<ushort> > array1, vector<vector<ushort> > array2)
{
	uint minDiff=0xEFFFFFF;
	int mi=-1,mj=-1;
	vector<ushort> result(3);

	for(uint i=0; i<array1.size(); i++)
	{
		for(uint j=0; j<array2.size(); j++)
		{
			uint diff = getHammingDiff( array1[i], array2[j] );
			//cout << i << "," << j << "="<< diff << endl;
			if(diff<minDiff)
			{
				minDiff=diff;
				mi=i;
				mj=j;
			}
		}
	}
	if(mi!=-1)
	{
		result[0]=mi;
		result[1]=mj;
		result[2]=(minDiff + 1 )*(array1[mi][3] + array2[mj][3]);
	}
	return result;
}

uint UtilCpp::getMaxDiffMedian(vector<vector<ushort> > array1, vector<vector<ushort> > array2)
{
	vector<ushort> difs;
	while(array1.size()!=0 && array2.size()!=0)
	{
		vector<ushort> dr = findMinDiff2(array1, array2);
		array1.erase(array1.begin()+dr[0]);
		array2.erase(array2.begin()+dr[1]);
		difs.push_back(dr[2]);
//		cout << dr[2] << endl;
	}
	uint medianDiffs = -1;
	if(difs.size()>0)
		medianDiffs=median(difs);
	return medianDiffs;
}

void UtilCpp::drawPoints(vector<Point> v, Mat & src, Scalar color)
{
	for(uint i=0; i<v.size(); i++)
	{
		circle(src,Point((int)v[i].x,(int)v[i].y),0.5,color,CV_FILLED);
//		Rect r((int)v[i].pt.x-2,(int)v[i].pt.y-2,5,5);
//		rectangle(src,r,color,0.5);
	}
}

void UtilCpp::drawKeyPoints(vector<KeyPoint> v, Mat & src, Scalar color)
{
	for(uint i=0; i<v.size(); i++)
	{
		circle(src,Point((int)v[i].pt.x,(int)v[i].pt.y),0.5,color,CV_FILLED);
//		Rect r((int)v[i].pt.x-2,(int)v[i].pt.y-2,5,5);
//		rectangle(src,r,color,0.5);
	}
}

vector<KeyPoint> UtilCpp::vecPoint2Keypoint(vector<Point> v, int kpsize) {

    vector<KeyPoint>  c_keypoints;

    for ( uint i = 0; i < v.size(); i++) {
    	c_keypoints.push_back(KeyPoint(v[i].x,v[i].y,kpsize));
    }
    return c_keypoints;
}

vector<Point> UtilCpp::vecKeyPoint2Point(vector<KeyPoint> v) {

    vector<Point>  points;

    for ( uint i = 0; i < v.size(); i++) {
    	points.push_back(Point(v[i].pt.x,v[i].pt.y));
    }
    return points;
}

void UtilCpp::susanEdges(Mat & src, int t)
{
	Mat result=Mat::zeros(src.size(),CV_8U);
	uchar *data = src.data;
	uchar *out = result.data;
	getSusanEdges(data,out,t,src.cols,src.rows);
	//return result;
	src=result;
}

void UtilCpp::susanPrincipal(Mat & src, int t)
{
	Mat result=Mat::zeros(src.size(),CV_8U);
	uchar *data = src.data;
	//uchar *out = result.data;
	getSusanPrincipal(data,t,src.cols,src.rows);
	//return result;
}

void UtilCpp::susanEdges3C(Mat & src, int t)
{
	//	#split image into component channels
	vector<Mat> planes;
	split(src, planes);

	susanEdges(planes[0],t);
	susanEdges(planes[1],t);
	susanEdges(planes[2],t);
	Mat s=Mat(src.size(),CV_32SC3);
	add(planes[0],planes[1],s);
	add(s,planes[2],s);
	src=s;
}

std::vector<float> UtilCpp::apply_permutation( std::vector<float> & vec, std::vector<int> & p)
{
	std::vector<float> sorted_vec(p.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
	return sorted_vec;
}

std::vector<int> UtilCpp::sort_permutation(vector<float> const & w)
{
	std::vector<int> p(w.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),[&](int i, int j){ return w[i]<w[j]; });
	return p;
}

std::vector<int> UtilCpp::apply_permutation( std::vector<int> & vec, std::vector<int> & p)
{
	std::vector<int> sorted_vec(p.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
	return sorted_vec;
}

std::vector<Size> UtilCpp::apply_permutation( std::vector<Size> & vec, std::vector<int> & p)
{
	std::vector<Size> sorted_vec(p.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](int i){ return vec[i]; });
	return sorted_vec;
}

std::vector<int> UtilCpp::sort_permutation(vector<int> const & w)
{
	std::vector<int> p(w.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),[&](int i, int j){ return w[i]<w[j]; });
	return p;
}

void UtilCpp::sortTwoVectors(vector<int> & src1, vector<int> & src2)
{
	// 			printVector(src1);
	// 			printVector(src2);
	vector <int> p = sort_permutation(src1);
	src1 = apply_permutation(src1, p);
	src2 = apply_permutation(src2, p);
	//			printVector(src1);
	//			printVector(src2);
}

vector<Point> UtilCpp::getComplementSet(vector<Point> v1, vector<Point> v2)
{
	sortstruct s;//(this);
	vector<Point> result;
	sortPointVec(v1);
	sortPointVec(v2);
	for(uint i=0; i<v1.size(); i++)
	{
		bool foundV1=false;
		for(uint j=0; j<v2.size(); j++)
		{
			if(v1[i].x==v2[j].x && v1[i].y==v2[j].y)
			{
				foundV1=true;
//				int x=(v2.begin()+j)->x;
//				int y=(v2.begin()+j)->y;
//				cout << "test: "<< x << "," << y << endl;
				v2.erase(v2.begin()+j);
				break;
			}
		}
		if(!foundV1)
			result.push_back(v1[i]);
		if(v2.size()==0)
		{
			result.insert(result.begin(),v1.begin()+i+1,v1.end());
			break;
		}
	}
//	for ( uint i = 0; i < result.size(); i++) {
//	    	cout << result[i].x << "," << result[i].y << endl;
//	    }
	return result;
};

vector<float> UtilCpp::subtract(vector<float> & data1, vector<float> & data2)
{
	vector<float> output;
	for (vector<float>::iterator it1 = data1.begin(), it2 = data2.begin();
			it1 != data1.end() && it2 != data2.end();
			++it1,  ++it2 )
	{
		output.push_back( *it1 - *it2 );
	}
	return output;
}

vector<Size> UtilCpp::getLargerGrid(Size g, Rect r, int scale)
{
	Size g1(g.width-1,g.height);
	Size g2(g.width,g.height-1);
	Size g3(g.width-2,g.height);
	Size g4(g.width,g.height-2);

	vector<Size> vs;
	vs.push_back(g);
	for(int i=g.width; i>=g.width-scale; i--)
	{
		for(int j=g.height; j>=g.height-scale; j--)
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
	vector<int> perm = sort_permutation(diffs);
//		u.printVector(diffs);
//		u.printVector(perm);

	vector<Size> orderedSizes=apply_permutation(vs,perm);
	//vector<Size> resultSizes;
	if(orderedSizes.size()>2*scale)
		orderedSizes.erase(orderedSizes.begin()+2*scale,orderedSizes.end());
	return orderedSizes;
}

Mat UtilCpp::proportionalMedianBlur(Mat & src)
{
	Mat tmp;
	tmp = src.clone();
	vector<Vec4i> hierarchy;
	vector<vector<Point> > contours;
	findContours( tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
	Mat mask=Mat::zeros(src.size(),CV_16U);
	Mat dest=Mat::zeros(src.size(),CV_16U);
	if(contours.size()<=0)
		return Mat();
	for( uint i = 0; i < contours.size(); i++ )
	{
		int area = contourArea(contours[i]);
		drawContours(mask,contours,i,Scalar(area),-1);
	}

	for(int i=0; i<src.rows; i++)
	{
		for(int j=0; j<src.cols; j++)
		{
			int ksize;
			int blobsize=mask.at<ushort>(i,j);
			if(blobsize!=0)
			{
				if(blobsize<100)
					ksize=13;
				else if(blobsize<700)
					ksize=3;
				else
					ksize=5;
				int hk=ksize/2;
				Rect r(abs(j-hk),abs(i-hk),MIN(j+hk,mask.cols),MIN(i+hk,mask.rows));
				Mat roi(src,r);
				std::vector<uchar> pixels(roi.rows * roi.cols);
				cv::Mat m(roi.rows, roi.cols, CV_8U, &pixels[0]);
				roi.copyTo(m);
				uchar med = median(pixels);
				dest.at<uchar>(i,j)=med;
			}
		}
	}
	return dest;
}

inline bool UtilCpp::exists_test (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}


bool operator==(const Point2f& pt1, const Point2f& pt2)
{
    return ((pt1.x == pt2.x) && (pt1.y == pt2.y));
}

namespace std
{
    template<>
    struct hash<Point2f>
    {
        size_t operator()(Point2f const& pt) const
        {
            return (size_t)(pt.x*100 + pt.y);
        }
    };
}


void UtilCpp::removedupes(std::vector<Point2f> & vec)
{
    std::unordered_set<Point2f> pointset;

    auto itor = vec.begin();
    while (itor != vec.end())
    {
        if (pointset.find(*itor) != pointset.end())
        {
            itor = vec.erase(itor);
        }
        else
        {
            pointset.insert(*itor);
            itor++;
        }
    }
}


vector<float> normalize(vector<float> v)
{
	double S = std::accumulate(v.begin(), v.end(), 0.0); // the total weight
	std::transform(v.begin(), v.end(), v.begin(), std::bind1st(std::multiplies<float>(),1/S));
	return v;
}

vector<vector<int> > UtilCpp::getFeatureList(string csvfile)
{
	  io::CSVReader<8> in(csvfile);
	  //in.read_header(io::ignore_extra_column);
	  string f,b,m,g,i,j,k,l;
	  vector<vector<int> > fvector;
	  while(in.read_row(f,b,m,g,i,j,k,l)){
	    vector<int> featureValues=vector<int>(8);
	    //cout << f << ","<< b<< "," << m << ","<< g << ","<< i << "," << j << "," << k << "," << l << endl;
	    if (isdigit(f[1]))
	    	featureValues[0]=(f[1] - '0');
	    if (isdigit(b[1]))
			featureValues[1]=(b[1] - '0');
	    if (isdigit(m[1]))
			featureValues[2]=(m[1] - '0');
	    if (isdigit(g[1]))
	    	featureValues[3]=(g[1] - '0');
	    if (isdigit(i[1]))
			featureValues[4]=(i[1] - '0');
	    if (isdigit(j[1]))
			featureValues[5]=(j[1] - '0');
	    if (isdigit(k[1]))
	    	featureValues[6]=(k[1] - '0');
	    if (isdigit(l[1]))
			featureValues[7]=(l[1] - '0');
	    fvector.push_back(featureValues);
	    //featureValues.clear();
	  }
	  return fvector;
}
