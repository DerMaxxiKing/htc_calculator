#!/bin/sh

cd ${0%/*} || exit 1    # run from this directory

# Source tutorial clean functions
. $WM_PROJECT_DIR/bin/tools/CleanFunctions

cleanCase

rm -f 0/cellToRegion

for region in shell tube solid
do
    rm -rf 0/${region}/cellToRegion constant/${region}/polyMesh
done

