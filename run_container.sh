#!/bin/sh

podman run --rm \
  -v ~/.aws:/root/.aws \
  -e HORSE_ID_DATA_ROOT="/data" \
  -e AWS_PROFILE=SystemAdministrator-517695827388 \
  --entrypoint python \
  horse-id-lambda-image \
  horse_id.py http://host.containers.internal:5000