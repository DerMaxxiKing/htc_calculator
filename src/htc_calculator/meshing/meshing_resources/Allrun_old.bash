#!/bin/sh

cd ${0%/*} || exit 1

. $WM_PROJECT_DIR/bin/tools/RunFunctions

# Create the initial block mesh and decompose
runApplication blockMesh
runApplication decomposePar -copyZero

# Run snappy without layers
foamDictionary system/snappyHexMeshDict -entry castellatedMesh -set on
foamDictionary system/snappyHexMeshDict -entry snap -set on
foamDictionary system/snappyHexMeshDict -entry addLayers -set off
runParallel snappyHexMesh -overwrite

# Convert the face zones into mapped wall baffles and split
runParallel createBaffles -overwrite
runParallel mergeOrSplitBaffles -split -overwrite
rm -rf processor*/constant/polyMesh/pointLevel

# Run snappy again to create layers
foamDictionary system/snappyHexMeshDict -entry castellatedMesh -set off
foamDictionary system/snappyHexMeshDict -entry snap -set off
foamDictionary system/snappyHexMeshDict -entry addLayers -set on
runParallel -a snappyHexMesh -overwrite

# Split the mesh into regions
runParallel splitMeshRegions -cellZones -defaultRegionName solid -overwrite

runApplication reconstructParMesh -allRegions -constant
