#! /usr/bin/ksh

for FILE in *.bmp ; do 
echo $FILE | substr( $0, 0, 4 )
done
