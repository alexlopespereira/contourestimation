#ifndef PARTICLEDEFS_H_
#define PARTICLEDEFS_H_

/* number of bins of HSV in histogram */
#define NH 10
#define NS 10
#define NV 10
#define NVV 128
#include <vector>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
typedef struct histogram {
//  vector<float> vhisto;
  float histo[NH*NS + NV];   /**< histogram array */
  int n;                     /**< length of histogram array */
} histogram;

typedef struct particle {
  float x;          /**< current x coordinate */
  float y;          /**< current y coordinate */
  float s;          /**< scale */
  float xp;         /**< previous x coordinate */
  float yp;         /**< previous y coordinate */
  float sp;         /**< previous scale */
  float x0;         /**< original x coordinate */
  float y0;         /**< original y coordinate */
  int width;        /**< original width of region described by particle */
  int height;       /**< original height of region described by particle */
  //histogram* histo; /**< reference histogram describing region being tracked */
  float histog[NH*NS + NV];
  float vhistos[NVV][NH*NS + NV];
  //vector<vector<float> > * vhistos;
  int n;
  float w;          /**< weight */
  Rect r;
  bool toEval;
  //int pid;
  particle(){
	  x=y=s=xp=yp=sp=x0=y0=width=height=n=w=0;
	  //histo=NULL;
  }
  particle(particle * p)
  {
	  x=p->x;
	  y=p->y;
	  s=p->s;
	  xp=p->xp;
	  yp=p->yp;
	  sp=p->sp;
	  x0=p->x0;
	  y0=p->y0;
	  width=p->width;
	  height=p->height;
	  //histo=p->histo;
	  w=p->w;
	  n=p->n;
	  r=Rect(p->r);
	  memcpy(&histog[0],&p->histog[0],n*4);
  }
} particle;



#endif
