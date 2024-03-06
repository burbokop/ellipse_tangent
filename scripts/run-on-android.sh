#!/bin/bash

set -e

PLATFORM_TOOLS_DIR=~/Android/Sdk/platform-tools
if test -d ${PLATFORM_TOOLS_DIR}; then
    export PATH=$PATH:$PLATFORM_TOOLS_DIR
fi

EXE="$1"
EXE_NAME=`basename $EXE`
adb push "$EXE" "/data/local/tmp/$EXE_NAME"
adb shell "chmod 755 /data/local/tmp/$EXE_NAME"

cmd="RUST_BACKTRACE=full RUST_LOG=debug /data/local/tmp/$EXE_NAME $2"
out=`adb shell "$cmd && echo SUCCESS || echo FAIL"`
echo "$out"
if [[ `echo "$out" | tail -n1` =~ ^FAIL  ]]; then
    exit 1
fi
