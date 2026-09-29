#!/bin/sh
# The asset path used by the engine is relative to its working directory, so
# the executable runs from the bundle root. Existing relative paths passed as
# arguments (`path` or `--option=path`) are resolved against the caller's
# directory first so they keep pointing at the same files.
bundle_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd) || exit 1

for arg do
    shift
    case $arg in
        --*=*)
            value=${arg#*=}
            case $value in
                /*) ;;
                *) if [ -e "$value" ]; then arg="${arg%%=*}=$PWD/$value"; fi ;;
            esac
            ;;
        /* | -*) ;;
        *) if [ -e "$arg" ]; then arg="$PWD/$arg"; fi ;;
    esac
    set -- "$@" "$arg"
done

cd "$bundle_dir" || exit 1
exec ./dirkengine "$@"
