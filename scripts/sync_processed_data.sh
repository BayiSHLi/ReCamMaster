#!/bin/bash

# ==============================
# Configurable Parameters
# ==============================

SOURCE_DIR="/mnt/hdd/dataset/webvid10m/data/"
TARGET_USER="root"
TARGET_HOST="10.97.40.240"
TARGET_DIR="/dataset/webvid10m/data/"

MODE="push"      # push | pull
DRY_RUN=false    # true | false

# ==============================
# Do not modify below
# ==============================

RSYNC_FLAGS="-avz --ignore-existing"

if [ "$DRY_RUN" = true ]; then
    RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
fi

if [ "$MODE" = "push" ]; then
    echo ">>> Pushing processed .pth files to $TARGET_HOST ..."
    
    rsync $RSYNC_FLAGS \
        "$SOURCE_DIR" \
        "${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}"

elif [ "$MODE" = "pull" ]; then
    echo ">>> Pulling processed .pth files from $TARGET_HOST ..."
    
    rsync $RSYNC_FLAGS \
        "${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}" \
        "$SOURCE_DIR"

else
    echo "Invalid MODE. Use push or pull."
    exit 1
fi

echo ">>> Sync complete."