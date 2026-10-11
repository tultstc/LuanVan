#!/usr/bin/env bash
# Download and prepare AVA v2.2 for projects/fgtn_ava (layout matches the config).
#
# Usage (from repo root):
#   bash projects/fgtn_ava/download_ava.sh [stage ...]
#
# Stages (default: all, in this order):
#   annotations  - AVA v2.2 annotation csv/pbtxt (~15 MB)
#   proposals    - FAIR dense person proposals pkl (~700 MB)
#   videos       - 299 trainval source videos (~160 GB, resumable)
#   cut          - trim each video to its 15th-30th minute (needs ffmpeg)
#   frames       - extract 30fps rawframes (~500 GB, needs ffmpeg)
#
# Env overrides:
#   DATA_DIR=data/ava   NUM_JOBS=4 (parallel video downloads)
#   SCALE=256           short side of extracted frames (0 = native resolution)
#   JPEG_Q=3            jpg quality, 1 (best/largest) .. 5 (smaller)
#   CLEAN=1             delete each source/cut video right after it is processed
#
# Small-disk subset (~25-30 GB instead of ~700 GB): after `annotations proposals`,
# run `python projects/fgtn_ava/make_subset.py` — the videos stage then only
# downloads the subset list, and training uses the *_subset annotation files.
#
# URLs mirror mmaction2's tools/data/ava scripts.

set -euo pipefail

DATA_DIR=${DATA_DIR:-"data/ava"}
NUM_JOBS=${NUM_JOBS:-4}
SCALE=${SCALE:-256}
JPEG_Q=${JPEG_Q:-3}
CLEAN=${CLEAN:-0}
ANNO_DIR="${DATA_DIR}/annotations"
VIDEO_DIR="${DATA_DIR}/videos"
CUT_DIR="${DATA_DIR}/videos_15min"
FRAME_DIR="${DATA_DIR}/rawframes"

# fetch <url> <dest_dir> — resumable download via wget or curl, whichever exists
fetch() {
    local url=$1 dir=$2
    if command -v wget >/dev/null; then
        wget -c -q --show-progress "${url}" -P "${dir}"
    else
        curl -fL -C - --progress-bar -o "${dir}/$(basename "${url}")" "${url}"
    fi
}

stage_annotations() {
    echo "==> [annotations] AVA v2.2 -> ${ANNO_DIR}"
    mkdir -p "${ANNO_DIR}"
    fetch https://research.google.com/ava/download/ava_v2.2.zip "${DATA_DIR}"
    unzip -oj "${DATA_DIR}/ava_v2.2.zip" -d "${ANNO_DIR}/"
    rm -f "${DATA_DIR}/ava_v2.2.zip"
    fetch https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt "${ANNO_DIR}"
}

stage_proposals() {
    echo "==> [proposals] FAIR dense proposals -> ${ANNO_DIR}"
    mkdir -p "${ANNO_DIR}"
    for f in train val; do
        fetch "https://download.openmmlab.com/mmaction/dataset/ava/ava_dense_proposals_${f}.FAIR.recall_93.9.pkl" \
            "${ANNO_DIR}"
    done
}

stage_videos() {
    local list="${ANNO_DIR}/ava_file_names_trainval_v2.1.txt"
    if [[ -f "${ANNO_DIR}/ava_file_names_subset.txt" ]]; then
        list="${ANNO_DIR}/ava_file_names_subset.txt"
        echo "==> [videos] subset list detected: $(wc -l < "${list}") videos"
    else
        echo "==> [videos] full set: 299 videos (~160 GB)"
    fi
    [[ -f "${list}" ]] || { echo "missing ${list} — run the annotations stage first"; exit 1; }
    mkdir -p "${VIDEO_DIR}"
    export -f fetch
    export VIDEO_DIR
    xargs -P "${NUM_JOBS}" -I{} bash -c 'fetch "https://s3.amazonaws.com/ava-dataset/trainval/$1" "${VIDEO_DIR}"' _ {} < "${list}"
    echo "downloaded: $(ls "${VIDEO_DIR}" | wc -l)/$(wc -l < "${list}")"
}

stage_cut() {
    command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
    echo "==> [cut] minute 15-30 of each video -> ${CUT_DIR}"
    mkdir -p "${CUT_DIR}"
    local total done=0
    total=$(ls "${VIDEO_DIR}" | wc -l)
    for video in "${VIDEO_DIR}"/*; do
        done=$((done + 1))
        local out_name="${CUT_DIR}/${video##*/}"
        if [[ ! -f "${out_name}" ]]; then
            echo "  [${done}/${total}] cutting ${video##*/} (re-encodes 15 min, takes a few minutes)"
            # note: harmless h264 warnings like 'mmco: unref short failure' are expected on AVA videos
            ffmpeg -nostdin -ss 900 -t 901 -i "${video}" -r 30 -strict experimental "${out_name}" \
                -loglevel error -stats
        fi
        if [[ "${CLEAN}" == "1" ]]; then rm -f "${video}"; fi
    done
}

stage_frames() {
    command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
    local vf=()
    if [[ "${SCALE}" != "0" ]]; then
        # resize short side to SCALE (the training pipeline resizes to 256 anyway)
        vf=(-vf "scale='if(gt(a,1),-2,${SCALE})':'if(gt(a,1),${SCALE},-2)'")
    fi
    echo "==> [frames] 30fps jpg (short side ${SCALE}, q:v ${JPEG_Q}) -> ${FRAME_DIR}"
    mkdir -p "${FRAME_DIR}"
    local total done=0
    total=$(ls "${CUT_DIR}" | wc -l)
    for video in "${CUT_DIR}"/*; do
        done=$((done + 1))
        local video_name="${video##*/}"
        video_name="${video_name%.*}"
        local out_video_dir="${FRAME_DIR}/${video_name}"
        if [[ -d "${out_video_dir}" && -n "$(ls -A "${out_video_dir}" 2>/dev/null)" ]]; then
            continue  # already extracted
        fi
        echo "  [${done}/${total}] extracting ${video_name}"
        mkdir -p "${out_video_dir}"
        ffmpeg -nostdin -i "${video}" -r 30 "${vf[@]}" -q:v "${JPEG_Q}" \
            "${out_video_dir}/img_%05d.jpg" -loglevel error -stats
        if [[ "${CLEAN}" == "1" ]]; then rm -f "${video}"; fi
    done
}

STAGES=("$@")
[[ ${#STAGES[@]} -eq 0 ]] && STAGES=(annotations proposals videos cut frames)

for stage in "${STAGES[@]}"; do
    case "${stage}" in
        annotations) stage_annotations ;;
        proposals) stage_proposals ;;
        videos) stage_videos ;;
        cut) stage_cut ;;
        frames) stage_frames ;;
        *) echo "unknown stage: ${stage} (annotations|proposals|videos|cut|frames)"; exit 1 ;;
    esac
done

echo "done. Expected layout for the config:"
echo "  ${ANNO_DIR}/ava_{train,val}_v2.2.csv + excluded_timestamps + pbtxt + proposals pkl"
echo "  ${FRAME_DIR}/<video_id>/img_%05d.jpg"
