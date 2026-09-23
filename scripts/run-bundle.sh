#!/bin/sh
# The asset path used by the engine is relative to its working directory.
cd "$(dirname "$0")" || exit 1
exec ./dirkengine "$@"
