/*
 * Copyright(c) 2009-2011, Diego Schmaedech Martins (UFSM, Federal University of Santa Maria, Brazil).
 *
 *
 * All rights reserved.
 *
 * COMMERCIAL USE:
 * This library is part of Mothorus Eye Tracker developed under
 *                  GNU LESSER GENERAL PUBLIC LICENSE
 *                   Version 3, 29 June 2007 License
 * If you have any commercial interest in this work please contact schmadech@gmail.com
 *
 *


           IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install, copy or use the software.


                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
 *
 */

#define CV_NO_BACKWARD_COMPATIBILITY
#define N(x)(sizeof(x)/sizeof*(x))
 
 #include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
/*

int clh(IplImage *img){

int i,j,t=0,u=10;
uchar *datac;

IplImage *convert=cvCreateImage( cvGetSize(img), 8, 3 );
datac = (uchar *)convert->imageData;
cvCvtColor(img,convert,CV_BGR2HSV);

for(i=0;i< (convert->height);i++)
    for(j=0;j<(convert->width);j++){
        t=datac[i*convert->widthStep+j*convert->nChannels+0]+u;
        if(t>255)
            datac[i*convert->widthStep+j*convert->nChannels+0]=255;
        else
            datac[i*convert->widthStep+j*convert->nChannels+0]=u;
    }

cvCvtColor(convert, img, CV_HSV2BGR);
cvSaveImage("result.jpg",img,0);

}
*/

CvRect f_roi, le_roi, re_roi,n_roi , lie_roi, rie_roi;
CvPoint f_cp,l_cp, r_cp, n_cp;
void doMorphology(IplImage* img){
        //printf(" \n"  );
	IplImage *img_temp;
	//cvThreshold( img, img, 200, 255, CV_THRESH_BINARY );

	CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* contours = 0;
        //cvSmooth( img, img, CV_GAUSSIAN, 3, 3 , 0, 0);
        //cvNormalize( img, img, 0, 255, CV_MINMAX, 0);
	img_temp= cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);

        cvDilate( img, img, NULL, 1 );
        cvErode( img, img, NULL, 1 );
        cvMorphologyEx(img, img, img_temp, NULL, CV_MOP_OPEN, 1);
        cvCanny(img,img,240,255,3);

	//cvMorphologyEx(img, img, img_temp, NULL, CV_MOP_OPEN, 1);


    //cvNormalize( gray, gray, 0, 255, CV_MINMAX, 0);
    //cvSmooth( gray, gray, CV_GAUSSIAN, 3, 3 , 0, 0);
	cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );
        cvZero(img);
        if(CV_IS_SEQ(contours)){
            contours = cvApproxPoly( contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 1, 1 );
            if( contours->total > 0 ){
                for( ;contours; contours = contours->h_next ){
                    //if( contours->total <  10 )
                      //  continue;

                    //cvDrawContours( img, contours, CV_RGB(255,255,255), CV_RGB(255,255,255), 1, 1, CV_AA, cvPoint(0,0) );
                    //MouthContours::TeethArcLength = cvArcLength(  contours, CV_WHOLE_SEQ, -1);
                    //MouthContours::TeethAreaContour = cvContourArea( contours, CV_WHOLE_SEQ);
                    double area = cvContourArea( contours, CV_WHOLE_SEQ);
                    double arc = cvArcLength(  contours, CV_WHOLE_SEQ, -1);
                    double raio = arc/(2*CV_PI);
                    double circulariry = ((4*CV_PI*area)/(arc*arc)) ;
                   // printf(" (%g)\n", fabs(circulariry));

                    if(fabs(circulariry) > 0.5){
                        CvRect box = cvBoundingRect( contours,0);
                        //cvRectangle( img, cvPoint(box.x, box.y), cvPoint(box.x+box.width, box.y+box.height), CV_RGB(255,255,255),1, 8,0);
                        cvCircle(img, cvPoint(box.x+box.width/2, box.y+box.height/2),raio,CV_RGB(255,255,255), -1, 8, 0 );
                        /*
                        int i;
                        for( i=1; i<=contours-> total; i++ ){

                                CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, contours, 1 );

                               // printf(" (%d,%d)\n", p->x, p->y );
                                cvCircle(img, cvPoint(p->x,p->y),raio,CV_RGB(255,255,255), -1, 8, 0 );
                        }
*/
                    }
                }
            }
        }

         cvDilate( img, img, NULL, 1 );

}
int min(int a[],int n){int i,m=256;for(i=0;i<n;i++){if(a[i]<m)m=a[i];}return m;}
int max(int a[],int n){int i,m=0;for(i=0;i<n;i++){if(a[i]>m)m=a[i];}return m;}
int linearSeach(int a[],int k,int n){int i;for(i=0;i<n;i++)if(a[i]==k)return i;return -1;}
int calibrationDiameter(IplImage* img,int x,int y,int w,int h,int *inc) {
    int i=0,j=0,as[w],ad[w-1],ad_0[cvRound(w/2)-1],ad_1[cvRound(w/2)-1] ;
    uchar* d = (uchar *)img->imageData;
    for(i=0;i<img->width;i++) {
        if(i>=x&&i<(x+w)){
            int s = 0;
            for(j=0;j<img->height;j++){
                if(j>=y&&j<(y+h)){
                    s += d[j*img->widthStep/sizeof(uchar)+i];
                }
            }
            as[i-x] = cvRound(s/h);
        }
    }
    for(i=0;i<w-1;i++){
        ad[i]=as[i+1]-as[i];
        if(i < (cvRound(w/2)-1))
            ad_0[i]=as[i+1]-as[i];
        else
            ad_1[i-(cvRound(w/2)-1)]=as[i+1]-as[i];
    }
    int d0 = linearSeach(ad_0, min(ad_0,N(ad_0)), N(ad_0));
    int d1 = linearSeach(ad_1, max(ad_1,N(ad_1)), N(ad_1));
    if(d0 != -1 && d1 != -1){
        *inc = d0+((d1+N(ad_0))-d0)/2;
        cvLine(img,cvPoint(*inc,h/2),cvPoint(*inc,h/2),CV_RGB(255,255,255), 3, 8, 0 );
        cvLine(img,cvPoint(d0,h/2),cvPoint(d1+N(ad_0),h/2),CV_RGB(255,255,255), 1, 8, 0 );
        if(((d1+N(ad_0))-d0)<0)
            fprintf( stderr, "ERROR\n");
        return (d1+N(ad_0))-d0;
    }else{
        return -1;
        fprintf( stderr, "ERROR\n");
    }
}
void bubbleSort(int a[],int n){
  int i, j, t;
  for (i = n-1; i >= 0; i--){
      for (j = 1; j <= i; j++){ if (a[j-1] > a[j]){t = a[j-1];a[j-1] = a[j];a[j] = t;} }
  }
}
void detectEyes( IplImage* src, IplImage* copy, IplImage* gray ){
    CvMemStorage* le_storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade* le_cascade = (CvHaarClassifierCascade*)cvLoad( "f2.xml", 0, 0, 0 );
    CvMemStorage* re_storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade* re_cascade = (CvHaarClassifierCascade*)cvLoad( "f3.xml", 0, 0, 0 );
    CvMemStorage* n_storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade* n_cascade = (CvHaarClassifierCascade*)cvLoad( "f4.xml", 0, 0, 0 );
    cvClearMemStorage( le_storage );
    if( le_cascade ) {
        cvSetImageROI( copy, le_roi );
        CvSeq* leye = cvHaarDetectObjects( copy, le_cascade, le_storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT, cvSize(18,12) );
        if( leye->total != 0) {
            CvRect* r = (CvRect*)cvGetSeqElem( leye, 0 );
            cvRectangle( src, cvPoint(le_roi.x + r->x, le_roi.y + r->y), cvPoint(le_roi.x + r->x + r->width, le_roi.y + r->y + r->height), CV_RGB(0,0,255), 1, 8, 0 );
            //(xo=w-z-t)
            lie_roi = cvRect( le_roi.x + r->x +(r->width - r->width/2 - r->width/6), le_roi.y + ( r->y + r->height/2),  r->width/2,  r->height/8);
            l_cp = cvPoint(lie_roi.x, lie_roi.y+lie_roi.height/2);
            cvRectangle( src, cvPoint(lie_roi.x, lie_roi.y), cvPoint(lie_roi.x + lie_roi.width, lie_roi.y + lie_roi.height), CV_RGB(255,255,255), 1, 8, 0 );
        }else{
            fprintf( stdout, "ERROR|%d\n",3 );//nao encontrou arquivo olho esquerdo
        }
        cvResetImageROI(copy);
        if(CV_IS_STORAGE(le_storage)){
            cvReleaseMemStorage( &le_storage );
        }
        cvReleaseHaarClassifierCascade( &le_cascade );
    }else{
        fprintf( stdout, "ERROR|%d\n",2 );//nao encontrou arquivo f2.xml
    }

    cvClearMemStorage( re_storage );
    if( re_cascade ) {
        cvSetImageROI( copy, re_roi );
        CvSeq* reye = cvHaarDetectObjects( copy, re_cascade, re_storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT, cvSize(18,12) );
        if( reye->total != 0) {
            CvRect* r = (CvRect*)cvGetSeqElem( reye, 0 );
            cvRectangle( src, cvPoint(re_roi.x + r->x, re_roi.y + r->y), cvPoint(re_roi.x + r->x + r->width, re_roi.y + r->y + r->height), CV_RGB(0,0,255), 1, 8, 0 );
            rie_roi = cvRect(re_roi.x + r->x + r->width/4 , re_roi.y + ( r->y + r->height/2),  r->width/2, r->height/8);
            r_cp = cvPoint(rie_roi.x, rie_roi.y+rie_roi.height/2);
            cvRectangle( src, cvPoint(rie_roi.x, rie_roi.y), cvPoint(rie_roi.x + rie_roi.width, rie_roi.y + rie_roi.height), CV_RGB(255,255,255), 1, 8, 0 );
        }else{
            fprintf( stdout, "ERROR|%d\n",5 );//nao encontrou arquivo olho direito
        }
        cvResetImageROI(copy);
        if(CV_IS_STORAGE(re_storage)){
            cvReleaseMemStorage( &re_storage );
        }
        cvReleaseHaarClassifierCascade( &re_cascade );
    }else{
        fprintf( stdout, "ERROR|%d\n",4 );//nao encontrou arquivo f3.xml
    }

    cvClearMemStorage( n_storage );
    if( n_cascade ) {
        cvSetImageROI( copy, n_roi );
        CvSeq* rn = cvHaarDetectObjects( copy, n_cascade, n_storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT, cvSize(18, 15) );
        if( rn->total != 0) {
            CvRect* r = (CvRect*)cvGetSeqElem( rn, 0 );
            cvRectangle( src, cvPoint(n_roi.x + r->x, n_roi.y + r->y), cvPoint(n_roi.x + r->x + r->width, re_roi.y + r->y + r->height), CV_RGB(255,0,0), 1, 8, 0 );
            n_cp = cvPoint(n_roi.x + r->x+r->width/2, n_roi.y + r->y+r->height/2);
        }else{
            n_cp = cvPoint(f_cp.x, f_cp.y);
            fprintf( stdout, "ERROR|%d\n",6 );//nao encontrou arquivo nariz
        }
        cvResetImageROI(copy);
        if(CV_IS_STORAGE(n_storage)){
            cvReleaseMemStorage( &n_storage );
        }
        cvReleaseHaarClassifierCascade( &n_cascade );
    }else{
        fprintf( stdout, "ERROR|%d\n",7 );//nao encontrou arquivo f4.xml
    }
    //meanshift............

    IplImage* copy_mask = cvCreateImage( cvSize(src->width,src->height), 8, 3 );
    cvCopy(copy,copy_mask,0);
    cvSetImageROI( copy_mask, lie_roi );
    cvSetImageROI( src, lie_roi );
    cvPyrMeanShiftFiltering( src,copy_mask, 1, 10, 2, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,10,1));
    //cvSaveImage("mean.jpg" ,src, 0);

    //cvShowImage( "ml", copy_mask );
    cvResetImageROI(copy_mask);
    cvResetImageROI(src);

    cvSetImageROI( copy_mask, rie_roi );
    cvSetImageROI( src, rie_roi );
    cvPyrMeanShiftFiltering(  src,copy_mask, 1, 10, 2, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,10,1));
    //cvSaveImage("mean.jpg" ,src, 0);

    //cvShowImage( "mr", copy_mask );
    cvResetImageROI(copy_mask);
    cvResetImageROI(src);

    cvCvtColor(copy, gray, CV_BGR2GRAY);
    cvNormalize( gray, gray, 0, 255, CV_MINMAX, 0);
    CvScalar avg_gray = cvAvg(gray,0);
    IplImage* mul_image = cvCreateImage( cvSize(src->width,src->height), 8, 1 );
    cvSet(mul_image, avg_gray, 0);
    IplImage* lmask = cvCreateImage( cvSize(src->width,src->height), 8, 1 );
    IplImage* rmask = cvCreateImage( cvSize(src->width,src->height), 8, 1 );
    cvAdd(gray,gray,gray,0);
    cvCopy(gray, lmask, 0);
    cvCopy(gray, rmask, 0);
     //start left search
    cvSetImageROI( gray, lie_roi );
    cvSetImageROI( lmask, lie_roi );
    cvSetImageROI( mul_image, lie_roi );
    doMorphology(lmask);
    cvAndS(lmask, avg_gray, lmask,0);
    cvSub(gray,lmask,gray,0);
    cvSmooth( gray, gray, CV_GAUSSIAN, 3, 3 , 0, 0);
    int inc;
    int l_clib = calibrationDiameter(gray, lie_roi.x, lie_roi.y ,lie_roi.width, lie_roi.height, &inc);
    l_cp.x += inc;
   // findCircles(gray);
    cvSaveImage("left.jpg" ,gray, 0);
    cvShowImage( "l", gray );
    //cvShowImage( "maskl", lmask );
    cvResetImageROI(gray);
    cvResetImageROI(lmask);
    //end left search

    //start right search
    cvSetImageROI( gray, rie_roi );
    cvSetImageROI( rmask, rie_roi );
    doMorphology(rmask);
    cvAndS(rmask, avg_gray, rmask,0);
    cvSub(gray,rmask,gray,0);
    cvSmooth( gray, gray, CV_GAUSSIAN, 3, 3 , 0, 0);
    int r_clib = calibrationDiameter(gray, rie_roi.x, rie_roi.y ,rie_roi.width, rie_roi.height, &inc);
    r_cp.x += inc;
   // findCircles(gray);
    cvSaveImage("right.jpg" ,gray, 0);
    cvShowImage( "r", gray );
    //cvShowImage( "maskr", rmask );
    cvResetImageROI(gray);
    cvResetImageROI(rmask);
    //end right search

    cvCircle(src,l_cp,2,CV_RGB(255,0,0), 1, 8, 0);
    cvCircle(src,r_cp,2,CV_RGB(255,0,0), 1, 8, 0);
    cvCircle(src,n_cp,2,CV_RGB(255,0,0), 1, 8, 0);
    cvCircle(src,f_cp,2,CV_RGB(255,0,0), 2, 8, 0);
    float l_dist,r_dist;
    if(l_clib>0){
        l_dist= (f_cp.x-l_cp.x)*(12.0f/l_clib);
    }else{
        l_clib = lie_roi.height;
        l_dist= (f_cp.x-(lie_roi.x+lie_roi.width/2))*(12.0f/lie_roi.height);
    }
    if(r_clib>0){
        r_dist= (r_cp.x-f_cp.x)*(12.0f/r_clib);
    }else{
        r_clib = rie_roi.height;
        r_dist= ((rie_roi.x+rie_roi.width/2)-f_cp.x)*(12.0f/rie_roi.height);
    }
   float m_clib = (r_clib+l_clib)*0.5;
   fprintf(stdout,"fx|fy|lx|ly|rx|ry|nx|ny|lr|rr|ld|rd:%d|%d|%d|%d|%d|%d|%d|%d|%f|%f|%f|%f\n",f_cp.x,f_cp.y,l_cp.x,l_cp.y,r_cp.x,r_cp.y,n_cp.x,n_cp.y, 12.0f/l_clib,12.0f/r_clib,(f_cp.x-l_cp.x)*(12.0f/m_clib),(r_cp.x-f_cp.x)*(12.0f/m_clib));

    fprintf( stdout, "fx|fy|lx|ly|rx|ry|nx|ny|lr|rr|ld|rd:%d|%d|%d|%d|%d|%d|%d|%d|%f|%f|%f|%f\n",f_cp.x,f_cp.y,l_cp.x,l_cp.y,r_cp.x,r_cp.y,n_cp.x,n_cp.y, 12.0f/l_clib,12.0f/r_clib,l_dist,r_dist);

}
void detectFace( IplImage* src, IplImage* copy, IplImage* gray){
    CvMemStorage* f_storage = cvCreateMemStorage(0);
    CvHaarClassifierCascade* f_cascade = (CvHaarClassifierCascade*)cvLoad( "f1.xml", 0, 0, 0 );
    cvClearMemStorage( f_storage );
    if( f_cascade ) {
        CvSeq* faces = cvHaarDetectObjects( copy, f_cascade, f_storage, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT, cvSize(20, 20) );
        if( faces->total != 0) {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );
            f_roi =  cvRect(r->x, r->y + r->height/6, r->width,   r->height/3 + r->height/6);
            le_roi = cvRect(r->x, r->y + r->height/6, r->width/2, r->height/3 + r->height/6);
            re_roi = cvRect(r->x + r->width/2, r->y + r->height/6, r->width/2, r->height/3 + r->height/6);
            n_roi = cvRect(r->x + r->width/4, r->y + r->height/4, r->width/2, r->height/2);
            f_cp = cvPoint(r->x + r->width/2, r->y + r->height/2);
            detectEyes( src, copy, gray);

            cvRectangle( src, cvPoint(r->x,r->y), cvPoint(r->x + r->width,r->y + r->height), CV_RGB(255,0,0), 1, 8, 0 );
            cvRectangle( src, cvPoint(le_roi.x,le_roi.y), cvPoint(le_roi.x+le_roi.width, le_roi.y+le_roi.height), CV_RGB(0,255,0), 1, 8, 0 );
            cvRectangle( src, cvPoint(re_roi.x,re_roi.y), cvPoint(re_roi.x+re_roi.width, re_roi.y+re_roi.height), CV_RGB(0,255,0), 1, 8, 0 );

        }else{
            fprintf( stdout, "ERROR|%d\n", 1 );//nao encontrou a face
        }
        if(CV_IS_STORAGE(f_storage)){
            cvReleaseMemStorage( &f_storage );
        }
        cvReleaseHaarClassifierCascade( &f_cascade );
    }else{
        fprintf( stdout, "ERROR|%d\n", 0 );//nao encontrou arquivo f1.xml
    }


}

int main( int argc, char** argv ){


/*
 -->m0
docs/m0/Bruno-e/Bruno.jpg
docs/m0/Carlos-c/Carlos.jpg
docs/m0/Cristiano-c/Cristiano.jpg
docs/m0/Gabriela-c/Gabriela.jpg
docs/m0/Marcelo-c/Marcelo.jpg
-->m1
docs/m1/tmp-1094877146-e/tmp-1094877146.jpg
docs/m1/tmp-453634272-e/tmp-453634272.jpg
docs/m1/tmp-845170476-c/tmp-845170476.jpg
docs/m1/tmp-384452359-c/tmp-384452359.jpg
docs/m1/tmp-493223346-c/tmp-493223346.jpg
docs/m1/tmp-862274876-c/tmp-862274876.jpg
docs/m1/tmp-109693071-c/tmp-109693071.jpg
docs/m1/tmp-647599777-c/tmp-647599777.jpg
docs/m1/tmp-920430114-c/tmp-920430114.jpg
docs/m1/tmp-1117998239-c/tmp-1117998239.jpg
docs/m1/tmp-45276189-c/tmp-45276189.jpg
docs/m1/tmp-789687039-e/tmp-789687039.jpg

-->m2
docs/m2/tmp-1289348617-e/tmp-1289348617.jpg
docs/m2/tmp-1582287536-c/tmp-1582287536.jpg
docs/m2/tmp-1865484848-c/tmp-1865484848.jpg
docs/m2/tmp-2030975910-e/tmp-2030975910.jpg
docs/m2/tmp-1324256688-c/tmp-1324256688.jpg
docs/m2/tmp-1808493030-c/tmp-1808493030.jpg
docs/m2/tmp-1873451082-e/tmp-1873451082.jpg
docs/m2/tmp-1366167470-e/tmp-1366167470.jpg
docs/m2/tmp-1839280418-c/tmp-1839280418.jpg
docs/m2/tmp-2003127150-c/tmp-2003127150.jpg
*/

    argv[1] = "megan.jpg";

    cvNamedWindow( "occam", 1 );

    IplImage* src = cvLoadImage(argv[1], 1 );

    IplImage* copy = cvCreateImage( cvSize(src->width,src->height), 8, 3);
    IplImage* drw = cvCreateImage( cvSize(src->width,src->height), 8, 3 );
    IplImage* gray = cvCreateImage( cvSize(src->width,src->height), 8, 1 );
    //cvPyrDown(src,copy,CV_GAUSSIAN_5x5);
    //cvPyrUp(copy,src,CV_GAUSSIAN_5x5);
    cvCopy(src, copy, 0);
    cvCopy(src, drw, 0);
    //cvSmooth(copy, copy, CV_GAUSSIAN, 3, 3, 0, 0);

    if( src && copy && gray) {
        detectFace( src, copy, gray );
        cvShowImage( "occam", src );
        cvWaitKey(0);
        cvReleaseImage( &src );
        cvReleaseImage( &copy );
        cvReleaseImage( &gray );
    }else{
        fprintf( stdout, "ERROR|%d\n", 9 );//nao encontrou arquivo de imagem ou tipo de imagem nao eh valida
        return EXIT_FAILURE;
    }
    cvDestroyWindow("occam");
    return 0;
}

