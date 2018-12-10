#!/usr/bin/env sh
set -e

../bin/caffe.exe train \
    --solver=C:/caffe-master/train/solver.prototxt $@
