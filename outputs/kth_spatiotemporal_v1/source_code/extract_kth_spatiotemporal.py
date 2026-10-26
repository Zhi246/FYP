#!/usr/bin/env python3
"""Stream existing KTH videos at the OLD cached indices; extract 4x4 inputs.

Use the existing FYP_XZH NumPy1/OpenCV environment. No Torch dependency here.
The 4 time groups are VIDEO input encoding, not hardware-output time bins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REGIONS = [("top_left",0,32,0,32), ("top_right",0,32,32,64),
           ("bottom_left",32,64,0,32), ("bottom_right",32,64,32,64)]
TOLERANCE = 1e-7


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding="utf-8")


def encode_frames(frames):
    values=np.asarray(frames)
    if values.shape!=(17,64,64) or not np.issubdtype(values.dtype,np.floating):
        raise ValueError("Expected 17x64x64 normalized floating-point grayscale frames")
    if not np.isfinite(values).all() or values.min()<0 or values.max()>1:
        raise ValueError("Frames must be finite and normalized to [0,1]")
    differences=np.abs(np.diff(values,axis=0))
    temporal=differences.mean(axis=(1,2)).astype(np.float32)
    spatial=np.stack([differences[:,y0:y1,x0:x1].reshape(4,4,32,32).mean(axis=(1,2,3))
                      for _,y0,y1,x0,x1 in REGIONS]).astype(np.float32).reshape(16)
    return temporal,spatial


def independent_encode_u8(gray_u8):
    """Independent float64 nested-loop definition, not shared reduction code."""
    f=np.asarray(gray_u8,dtype=np.float64)/255.
    response=[]
    for _,y0,y1,x0,x1 in REGIONS:
        for group in range(4):
            interval_means=[]
            for i in range(group*4,group*4+4):
                interval_means.append(np.abs(f[i+1,y0:y1,x0:x1]-f[i,y0:y1,x0:x1]).mean())
            response.append(float(np.mean(interval_means)))
    temporal=np.array([np.abs(f[i+1]-f[i]).mean() for i in range(16)])
    return temporal,np.array(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/kth_cycle_robustness_v1"))
    parser.add_argument("--output",type=Path,default=Path("outputs/kth_spatiotemporal_v1/preprocessing"))
    args=parser.parse_args()
    args.source=args.source.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cv2.setNumThreads(1)
    manifest=pd.read_csv(args.source/"sample_manifest.csv")
    old=np.load(args.source/"inputs.npz",allow_pickle=False)
    assert manifest.sample_id.tolist()==old["sample_ids"].tolist()
    assert len(manifest)==2389 and manifest.sample_id.nunique()==2389
    hash_begin=time.perf_counter()
    videos=[]
    for row in manifest.drop_duplicates("path").itertuples():
        path=Path(row.path)
        if not path.is_file():raise FileNotFoundError(path)
        digest=sha256(path)
        if digest!=row.video_sha256:raise RuntimeError(f"Video bytes changed: {path}")
        videos.append({"path":str(path),"sha256":digest,"bytes":path.stat().st_size})
    hash_seconds=time.perf_counter()-hash_begin
    signature={"extractor_sha256":sha256(__file__),"manifest_sha256":sha256(args.source/"sample_manifest.csv"),
               "old_inputs_sha256":sha256(args.source/"inputs.npz"),"videos":videos,
               "opencv":cv2.__version__,"numpy":np.__version__,"feature_order":"region_major_then_time",
               "regions":REGIONS,"time_groups":[[1,4],[5,8],[9,12],[13,16]]}
    path=args.output/"signature.json"
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(signature)):
            raise RuntimeError("Extraction signature changed; preserve old cache and use a new --output")
    else:write_json(path,signature)
    if (args.output/"COMPLETE.json").is_file():
        print("Identical completed spatial-temporal cache reused; no video re-decoding.",flush=True)
        return
    positions=[]
    for r,(name,y0,y1,x0,x1) in enumerate(REGIONS):
        for group in range(4):
            positions.append({"input_position":r*4+group+1,"region":name,"y_start":y0,"y_end_exclusive":y1,
                "x_start":x0,"x_end_exclusive":x1,"time_group":group+1,
                "first_difference_index_one_based":group*4+1,"last_difference_index_one_based":group*4+4,
                "nominal_window_start_seconds":group*.25,"nominal_window_end_seconds":(group+1)*.25})
    pd.DataFrame(positions).to_csv(args.output/"input_dimension_definition.csv",index=False)
    examples=[]
    for cls in old["classes"]:
        subset=manifest[(manifest.split=="train")&(manifest["class"]==cls)].sort_values("sample_id").drop_duplicates("video_id").head(2)
        examples.extend(subset.index.tolist())
    example_set=set(examples)
    stored_frames={}; reconstructed=np.full((len(manifest),16),np.nan,dtype=np.float64)
    new=np.full_like(reconstructed,np.nan)
    failures=[]; records=[]; video_records=[]
    extraction_begin=time.perf_counter()
    for number,(video_id,part) in enumerate(manifest.groupby("video_id",sort=True),1):
        begin=time.perf_counter();capture=cv2.VideoCapture(part.iloc[0].path)
        plans={i:json.loads(row.sampled_indices) for i,row in part.iterrows()}
        wanted={frame for indices in plans.values() for frame in indices}
        frames={};timestamps={}
        if not capture.isOpened():
            reason="cannot_open_existing_video"
            failures.extend([{"sample_id":row.sample_id,"reason":reason} for row in part.itertuples()])
            capture.release();continue
        fps=float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or not np.allclose(part.fps,fps,rtol=0,atol=1e-8):
            raise RuntimeError(f"FPS differs from the saved sampling protocol: {video_id}")
        for index,row in part.iterrows():
            duration=(row.end_frame_1based-row.start_frame_1based+1)/fps
            targets=(row.start_frame_1based-1)/fps+(duration-1)/2+np.linspace(0,1,17)
            expected=[min(row.end_frame_1based-1,max(row.start_frame_1based-1,round(float(t*fps)))) for t in targets]
            assert expected==plans[index],f"Sampling plan changed for {row.sample_id}"
        for index in range(max(wanted)+1):
            if not capture.grab():break
            if index in wanted:
                okay,bgr=capture.retrieve()
                if okay and bgr is not None:
                    gray=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(64,64),interpolation=cv2.INTER_AREA)
                    frames[index]=gray
                    timestamps[index]=float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000.
        capture.release()
        for index,row in part.iterrows():
            indices=plans[index]
            missing=sorted(set(indices)-frames.keys())
            if missing:
                failures.append({"sample_id":row.sample_id,"reason":f"cannot_read_saved_indices:{missing}"})
                continue
            gray=np.stack([frames[i] for i in indices])
            temporal,spatial=encode_frames(gray.astype(np.float32)/255.)
            reconstructed[index]=temporal;new[index]=spatial
            old_error=float(np.max(np.abs(temporal-old["raw"][index])))
            conservation=float(np.max(np.abs(spatial.reshape(4,4).mean(axis=0)-temporal.reshape(4,4).mean(axis=1))))
            saved_timestamps=np.asarray(json.loads(row.decoder_timestamps_seconds))
            timestamp_error=float(np.max(np.abs(saved_timestamps-np.array([timestamps[i] for i in indices]))))
            records.append({"sample_id":row.sample_id,"old_temporal_max_abs_error":old_error,
                            "area_time_average_conservation_error":conservation,
                            "decoder_timestamp_max_error_seconds":timestamp_error,
                            "actual_sample_span_seconds":(indices[-1]-indices[0])/fps,
                            "repeated_frame_index_count":17-len(set(indices)),
                            "passed":old_error<=TOLERANCE and conservation<=TOLERANCE and timestamp_error<=TOLERANCE})
            if index in example_set:stored_frames[index]=gray
        video_records.append({"video_id":video_id,"sampled_unique_gray_frames":len(frames),
                              "maximum_retained_gray_bytes":len(frames)*64*64,"processing_seconds":time.perf_counter()-begin})
        if number%75==0 or number==manifest.video_id.nunique():
            print(f"Extracted spatial-temporal inputs: {number}/{manifest.video_id.nunique()} existing videos",flush=True)
    extraction_seconds=time.perf_counter()-extraction_begin
    pd.DataFrame(failures,columns=["sample_id","reason"]).to_csv(args.output/"new_extraction_failures.csv",index=False)
    pd.DataFrame(records).to_csv(args.output/"all_sample_reconstruction_checks.csv",index=False)
    pd.DataFrame(video_records).to_csv(args.output/"per_video_preprocessing.csv",index=False)
    if failures or len(records)!=len(manifest):
        raise RuntimeError("Sample set changed during extraction; old results cannot be reused. See new_extraction_failures.csv; no training is permitted.")
    assert np.isfinite(new).all() and (new>=0).all() and (new<=1).all()
    assert all(row["passed"] for row in records)
    assert np.max(np.abs(reconstructed-old["raw"]))<=TOLERANCE
    # Independent sequential read() decoder and float64 loop arithmetic on twelve
    # preselected TRAIN clips (two distinct source videos per class).
    audit_begin=time.perf_counter();audit=[]
    for index in examples:
        row=manifest.iloc[index];indices=json.loads(row.sampled_indices);wanted=set(indices)
        capture=cv2.VideoCapture(row.path);decoded={}
        for frame_index in range(max(indices)+1):
            okay,bgr=capture.read()
            if not okay:break
            if frame_index in wanted:
                decoded[frame_index]=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),(64,64),interpolation=cv2.INTER_AREA)
        capture.release()
        assert wanted<=decoded.keys()
        gray=np.stack([decoded[i] for i in indices])
        np.testing.assert_array_equal(gray,stored_frames[index])
        reference_temporal,reference_new=independent_encode_u8(gray)
        error_new=float(np.max(np.abs(reference_new-new[index])))
        error_old=float(np.max(np.abs(reference_temporal-old["raw"][index])))
        assert max(error_new,error_old)<=TOLERANCE
        audit.append({"sample_id":row.sample_id,"class":row["class"],"split":"train",
                      "independent_new_feature_max_abs_error":error_new,"independent_old_feature_max_abs_error":error_old,
                      "read_vs_grab_frames_exactly_equal":True,"tolerance":TOLERANCE,"passed":True})
    np.savez_compressed(args.output/"raw_features.npz",spatiotemporal=new,reconstructed_temporal=reconstructed,
                        sample_ids=old["sample_ids"],labels=old["labels"],splits=old["splits"])
    np.savez_compressed(args.output/"training_examples.npz",gray_frames_u8=np.stack([stored_frames[i] for i in examples]),
                        sample_ids=old["sample_ids"][examples],classes=manifest.loc[examples,"class"].to_numpy(dtype=str),
                        old_raw=old["raw"][examples],new_raw=new[examples],
                        sampled_indices=np.array([json.loads(manifest.iloc[i].sampled_indices) for i in examples]),
                        fps=manifest.iloc[examples].fps.to_numpy())
    shared_checks={"passed":True,"same_samples":len(manifest),"same_unique_videos":len(videos),
                   "all_source_video_bytes_unchanged":True,"new_exclusions":0,
                   "old_temporal_max_abs_error":float(np.max(np.abs(reconstructed-old["raw"]))),
                   "conservation_max_abs_error":max(r["area_time_average_conservation_error"] for r in records),
                   "timestamp_max_error_seconds":max(r["decoder_timestamp_max_error_seconds"] for r in records),
                   "fixed_sampling_plan_matches_cached_indices":True,"tolerance":TOLERANCE,
                   "independent_training_audit":audit,"new_shape":list(new.shape),
                   "new_all_zero_rows":int((new==0).all(axis=1).sum()),"new_min":float(new.min()),"new_max":float(new.max())}
    write_json(args.output/"extraction_checks.json",shared_checks)
    write_json(args.output/"cost_and_environment.json",{
        "wall_seconds":time.perf_counter()-started,"video_hashing_seconds":hash_seconds,
        "decode_and_feature_seconds":extraction_seconds,"independent_audit_and_cache_save_seconds":time.perf_counter()-audit_begin,
        "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "original_video_bytes_unique":sum(v["bytes"] for v in videos),"new_raw_float64_bytes":new.nbytes,
        "all_raw_video_frames_loaded_at_once":False,"maximum_retained_video_gray_bytes":max(v["maximum_retained_gray_bytes"] for v in video_records),
        "new_download_seconds":0.,"numpy":np.__version__,"opencv":cv2.__version__,"opencv_threads":cv2.getNumThreads()})
    write_json(args.output/"COMPLETE.json",{"completed":True,"same_samples":True,"feature_shape":[2389,16]})
    print(f"COMPLETE: {args.output}; old-feature maximum error {shared_checks['old_temporal_max_abs_error']:.3g}",flush=True)


if __name__=="__main__":main()
