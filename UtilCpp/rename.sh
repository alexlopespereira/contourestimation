#! /usr/bin/ksh
typeset -R5 val2
NUM=0;
for FILE in *.bmp ; do 
NUM = `echo $FILE | awk '{ substr( $0, 3, 5 ) }'`
echo $NUM
#NUM=`expr $NUM + 1` ; 
#val2="0000"$NUM
#val3=b$val2.bmp 
#echo $val3
#mv $FILE $val3
done
