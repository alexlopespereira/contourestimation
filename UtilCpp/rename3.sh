#!/bin/bash
indir=$1
ext=$2
currDir=$PWD
cd $indir
for FILE in `ls *.$ext` ; do 
	num=$(echo $FILE | cut -d. -f2)
	mv $FILE "in0$num.jpg"
done
cd $currDir
