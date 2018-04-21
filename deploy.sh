#!/usr/bin/env bash

# get run shell dictionary
SOURCE="$0"
while [ -h "$SOURCE"  ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /*  ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd  )"
echo $DIR

cd $DIR
# do update
git pull
#
sudo python3 setup.py install

PROCESS=`ps -ef|grep 8989|grep -v grep|grep -v PPID|awk '{ print $2}'`
for i in $PROCESS
do
   echo "Kill the $1 process [ $i ]"
   kill -9 $i
done

