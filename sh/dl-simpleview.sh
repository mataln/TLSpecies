#!/bin/bash

#Exception for if scripts are called from within the /sh/ directory
#This will break if there's a space in the path
if [$(basename $PWD) = "sh" ]
then
	cd ..
fi

#Clone Simpleview repo
git clone https://github.com/IsaacCorley/simpleview-pytorch

cd simpleview-pytorch

#Remove git stuff + non-classification bits
rm -r assets
rm -f LICENSE
rm -f README.md
rm -f .gitignore

cd ..

mv simpleview-pytorch/simpleview_pytorch simpleview_pytorch
rm -r -f simpleview-pytorch