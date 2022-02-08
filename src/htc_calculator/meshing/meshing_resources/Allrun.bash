#!/bin/sh

cd ${0%/*} || exit 1

. $WM_PROJECT_DIR/bin/tools/RunFunctions

rm -rf constant/polyMesh/sets

runApplication surfaceFeatures
runApplication blockMesh
runApplication snappyHexMesh -overwrite
runApplication splitMeshRegions -cellZones -overwrite
paraFoam -touchAll
