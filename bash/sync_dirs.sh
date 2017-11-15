#!/bin/sh

# requires having set environment variables for user and pass:
# IBOX_USER and IBOX_PASS

SOURCE=/home/pchtsp/Documents/projects/OPTIMA_documents/
DEST=//disc/f.peschiera/Documents/OPTIMA
TEMP_MOUNT=/mnt/Shared

if [ -n "`mount | grep $TEMP_MOUNT`" ]; then
    echo "Already mounted"
else
    echo "Not mounted: attempting to mount..."
    sudo mount "$DEST" "$TEMP_MOUNT" -o user="$IBOX_USER",password="$IBOX_PASS"
fi

if [ -n "`mount | grep $TEMP_MOUNT`" ]; then
    echo "Apparently, we manage to mount things..."
    echo "Now, we try to sync the directories..."
    sudo rsync -rvu --delete "$SOURCE" "$TEMP_MOUNT"
    echo "Finally, we try to unmount the connection"
    sudo umount "$TEMP_MOUNT"
else
    echo "Crap! Mount Failed :( "
fi