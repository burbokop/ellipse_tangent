#!/bin/sh

set -e

PLATFORM_TOOLS_DIR=~/Android/Sdk/platform-tools
if test -d ${PLATFORM_TOOLS_DIR}; then
    export PATH=$PATH:$PLATFORM_TOOLS_DIR
fi

EXE="$1"
EXE_NAME=`basename $EXE`
adb push "$EXE" "/data/local/tmp/$EXE_NAME"
adb shell "chmod 755 /data/local/tmp/$EXE_NAME"
OUT="$(mktemp)"
adb shell "RUST_BACKTRACE=full RUST_LOG=debug /data/local/tmp/$EXE_NAME $2" 2>&1 | tee $OUT
