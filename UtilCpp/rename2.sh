#! /usr/bin/ksh
typeset -R9 val2
NUM=0;
for FILE in *.bmp ; do 
NUM=`echo $FILE | tail -c 8`
val2="00"$NUM
val3=b$val2 
mv $FILE $val3
done
