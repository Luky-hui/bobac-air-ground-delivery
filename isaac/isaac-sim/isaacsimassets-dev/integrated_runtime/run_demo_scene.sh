#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DEMO_ROOT/../.." && pwd)"

ISAACSIM_ROOT=${ISAACSIM_ROOT:-/isaac-sim}
ISAACSIM_PATH=$ISAACSIM_ROOT
ISAACSIM_LAUNCHER=${ISAACSIM_LAUNCHER:-$ISAACSIM_ROOT/runheadless.sh}
ISAACSIM_SITE=$ISAACSIM_ROOT/python_packages
HUMBLE_LIB=$ISAACSIM_ROOT/exts/isaacsim.ros2.bridge/humble/lib
KIT_LIB=$ISAACSIM_ROOT/kit
PEGASUS_EXTENSION=${PEGASUS_EXTENSION:-/root/PegasusSimulator/extensions/pegasus.simulator}
NVIDIA_VK_ICD=${NVIDIA_VK_ICD:-/etc/vulkan/icd.d/nvidia_icd.json}
X1_ASSEMBLED_SCENE_USD=${X1_ASSEMBLED_SCENE_USD:-$PROJECT_ROOT/isaac_sim_2026/isaac-sim/X1_race_scene_new3.usd}

if [ ! -f "$NVIDIA_VK_ICD" ] && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  NVIDIA_VK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

WORLD_MODE=""
PASSTHROUGH_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --world)
      WORLD_MODE="${2:-}"
      PASSTHROUGH_ARGS+=("$1" "$WORLD_MODE")
      shift 2
      ;;
    --world=*)
      WORLD_MODE="${1#--world=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -n "$WORLD_MODE" ]; then
  case "$WORLD_MODE" in
    X1|Bobac|bobac|create_map)
      ;;
    *)
      echo "Unsupported --world '$WORLD_MODE'. Use X1, Bobac, or create_map." >&2
      exit 2
      ;;
  esac
fi

exec env -i \
  HOME=${HOME:-/root} \
  USER=${USER:-root} \
  LOGNAME=${LOGNAME:-root} \
  OMNI_KIT_ALLOW_ROOT=1 \
  PATH=$ISAACSIM_ROOT:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  DISPLAY=${DISPLAY:-} \
  XAUTHORITY=${XAUTHORITY:-} \
  NVIDIA_VISIBLE_DEVICES=all \
  ACCEPT_EULA=Y \
  PRIVACY_CONSENT=Y \
  VK_DRIVER_FILES=$NVIDIA_VK_ICD \
  VK_ICD_FILENAMES=$NVIDIA_VK_ICD \
  VK_LAYER_NV_optimus=NVIDIA_only \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 \
  LD_LIBRARY_PATH=$HUMBLE_LIB:$KIT_LIB:$ISAACSIM_ROOT/kit/kernel/py:${LD_LIBRARY_PATH:-} \
  PYTHONPATH=$DEMO_ROOT:$PEGASUS_EXTENSION:$ISAACSIM_ROOT/kit/kernel/py:$ISAACSIM_SITE:${PYTHONPATH:-} \
  ISAACSIM_PATH=$ISAACSIM_PATH \
  ISAACSIM_ROOT=$ISAACSIM_ROOT \
  PEGASUS_EXTENSION=$PEGASUS_EXTENSION \
  ROS_DISTRO=humble \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ISAACSIM_LAUNCHED=1 \
  CARGO_DELIVERY_WORLD=$WORLD_MODE \
  X1_ASSEMBLED_SCENE_USD=$X1_ASSEMBLED_SCENE_USD \
  $ISAACSIM_LAUNCHER --exec "$SCRIPT_DIR/scene_app.py" "${PASSTHROUGH_ARGS[@]}"
