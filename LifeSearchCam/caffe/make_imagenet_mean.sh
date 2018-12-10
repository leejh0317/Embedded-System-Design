#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=../data
DATA=../data
TOOLS=../bin

$TOOLS/compute_image_mean.exe $EXAMPLE/jw_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
