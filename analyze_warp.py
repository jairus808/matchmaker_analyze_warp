import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import pyaudio

from matchmaker import Matchmaker
#modules needed to import to access frame rate and hop length
from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    ChromagramProcessor,
    CQTProcessor,
    LogSpectralEnergyProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    SAMPLE_RATE,
)
#install python-osc
from pythonosc.udp_client import SimpleUDPClient 
from matchmaker.io.audio import AudioStream
from matchmaker.prob.hmm import GaussianAudioPitchHMM, GaussianAudioPitchTempoHMM


os.environ["PARTITURA_SOUNDFONT"] = "tests/Steinway_B_soundfont.sf2"

DEFAULT_SCORE = Path("matchmaker/assets/simple_score.musicxml")

#An unclean performance
DEFAULT_AUDIO_PERF = Path("matchmaker/assets/simple_performance.mp3")
DEFAULT_ANNOTS = Path("matchmaker/assets/simple_perf_annotations.txt")
DEFAULT_RESULTS_DIR = Path("results")


FRAME_RATE = 30 #fps, default used by matchmaker is 30, use 25 for simple_example.tsv if you 

#make pandas datafram with computed slope for easy printout functionality
def _add_temporal_columns(df: pd.DataFrame, frame_rate: int) -> pd.DataFrame:
    df["score_time"] = df["ref_idx"] / frame_rate
    df["perf_time"] = df["input_idx"] / frame_rate


#append to NaN instead of 0 to prevent. tempo_ratio 
    dt_score = np.diff(df["score_time"].to_numpy(), prepend=np.nan)
    dt_perf = np.diff(df["perf_time"].to_numpy(), prepend=np.nan)
    delta_ref = np.diff(df["ref_idx"].to_numpy(), prepend=np.nan)
    delta_perf = np.diff(df["input_idx"].to_numpy(), prepend=np.nan)

    delta_ratio = np.full_like(delta_perf, np.nan, dtype=np.float64)
    valid = ~np.isnan(delta_ref) & (delta_ref != 0)
    delta_ratio[valid] = delta_perf[valid] / delta_ref[valid]
    leading = (~valid) & (delta_perf > 0)
    delta_ratio[leading] = delta_perf[leading]

    df["dt_score"] = dt_score
    df["delta_ref"] = delta_ref
    df["delta_perf"] = delta_perf
    df["delta_ratio"] = delta_ratio
    df["perf_leading"] = (np.abs(dt_score) == 0) & (dt_perf > 0)
    # tempo_ratio mirrors delta_ratio only when the score advances; otherwise nan
    df["tempo_ratio"] = np.where(np.abs(delta_ref) > 0, delta_ratio, np.nan)
    return df


#account for nan
def _format_ratio(ratio: float) -> str:
    if ratio is None or np.isnan(ratio):
        return "nan"
    return f"{ratio:.4f}"


def _find_runs(mask: pd.Series) -> List[tuple[int, int, int]]:
    """Return list of (start, end, length) for contiguous True spans."""
    runs: List[tuple[int, int, int]] = []
    start = None
    for i, val in mask.reset_index(drop=True).items():
        if val and start is None:
            start = i
        if (not val or i == len(mask) - 1) and start is not None:
            end = i if val else i - 1
            runs.append((start, end, end - start + 1))
            start = None
    return runs


def summarize_runs(df: pd.DataFrame) -> None:
    """Print quick summaries of contiguous perf-leading and jump events."""
    lead_mask = (df["delta_ref"] == 0) & (df["delta_perf"] > 0)
    jump_mask = df["delta_ref"] >= 5
    lead_runs = _find_runs(lead_mask)
    jump_runs = _find_runs(jump_mask)
    max_lead = max((l for *_, l in lead_runs), default=0)
    max_jump = max((l for *_, l in jump_runs), default=0)
    print(
        f"[summary] perf_leading runs: {len(lead_runs)} (max len={max_lead})",
        flush=True,
    )
    print(
        f"[summary] score_jump (delta_ref>=5) runs: {len(jump_runs)} (max len={max_jump})",
        flush=True,
    )
    # emit run boundaries for downstream processing
    if lead_runs:
        print(
            "[summary] perf_leading boundaries (start,end,len): "
            + "; ".join(f"{s},{e},{l}" for s, e, l in lead_runs),
            flush=True,
        )
    if jump_runs:
        print(
            "[summary] score_jump boundaries (start,end,len): "
            + "; ".join(f"{s},{e},{l}" for s, e, l in jump_runs),
            flush=True,
        )


# run matchmakers  ref vs. input
def stream_from_file(
    tsv_path: Path,
    frame_rate: int,
    playback: bool = False,
    performance_path: Optional[Path] = None,
    wait: bool = True,
    osc: Optional["SimpleUDPClient"] = None,
) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["ref_idx", "input_idx"],
    )
    df = _add_temporal_columns(df, frame_rate)
    print(
        "[file] Loaded "
        f"{len(df)} frames (max ref_idx={int(df['ref_idx'].max())}, "
        f"max input_idx={int(df['input_idx'].max())})",
        flush=True,
    )

    stop_event: Optional[threading.Event] = None
    playback_thread: Optional[threading.Thread] = None
    lead_run_start: Optional[int] = None
    jump_run_start: Optional[int] = None
    if playback:
        if performance_path is None:
            print("[file] Playback requested but no performance file provided.")
        elif not performance_path.exists():
            print(f"[file] Playback requested but file not found: {performance_path}")
        else:
            stop_event = threading.Event()
            playback_thread = _start_playback(
                performance_path=performance_path, stop_event=stop_event
            )

    try:
        for idx, row in df.iterrows():
            ratio_val = row["tempo_ratio"]
            delta_ratio_val = row.get("delta_ratio", np.nan)
            ratio_str = _format_ratio(ratio_val)
            grad_str = _format_ratio(delta_ratio_val)
            delta_ref_str = _format_ratio(row.get("delta_ref", np.nan))
            delta_perf_str = _format_ratio(row.get("delta_perf", np.nan))
            dt_score_str = _format_ratio(row["dt_score"])
            print(
                f"[file] Frame {idx:05d} | score_idx={int(row['ref_idx'])} | "
                f"perf_idx={int(row['input_idx'])} | "
                f"Δref={delta_ref_str} | Δperf={delta_perf_str} | "
                f"t_r={ratio_str} | grad={grad_str} | dt_score={dt_score_str}",
                flush=True,
            )

            if osc:
                osc.send_message(
                    "/warp",
                    [
                        int(row["ref_idx"]),
                        int(row["input_idx"]),
                        float(row.get("delta_ratio", np.nan)),
                        float(row.get("tempo_ratio", np.nan)),
                        float(row.get("dt_score", np.nan)),
                        float(row.get("delta_ref", np.nan)),
                        float(row.get("delta_perf", np.nan)),
                        int(row.get("perf_leading", False)),
                    ],
                )

            # event logging as soon as runs end. Had to take from a stack overflow discussion that I accessed from chatgpt (contiguous events on pandas ..
            #.https://stackoverflow.com/questions/65238399/how-to-get-start-and-end-indices-of-consecutive-groups-of-data-in-pandas?utm_source=chatgpt.com)
            lead_now = (row.get("delta_ref", np.nan) == 0) and (row.get("delta_perf", np.nan) > 0)
            jump_now = row.get("delta_ref", 0) >= 5 if not np.isnan(row.get("delta_ref", np.nan)) else False
            if lead_now and lead_run_start is None:
                lead_run_start = idx
            if not lead_now and lead_run_start is not None:
                end = idx - 1
                length = end - lead_run_start + 1
                print(f"[event] perf_leading run start={lead_run_start} end={end} len={length}", flush=True)
                if osc:
                    osc.send_message("/event", ["perf_leading", lead_run_start, end, length])
                lead_run_start = None

            if jump_now and jump_run_start is None:
                jump_run_start = idx
            if not jump_now and jump_run_start is not None:
                end = idx - 1
                length = end - jump_run_start + 1
                print(f"[event] score_jump run start={jump_run_start} end={end} len={length}", flush=True)
                if osc:
                    osc.send_message("/event", ["score_jump", jump_run_start, end, length])
                jump_run_start = None

            #wait boolean logic...
            if wait and frame_rate > 0:
                time.sleep(1 / frame_rate)
    finally:
        if playback_thread is not None:
            # Let playback finish naturally; join so the audio completes
            playback_thread.join()

    # flush any open runs
    if lead_run_start is not None:
        end = len(df) - 1
        length = end - lead_run_start + 1
        print(f"[event] perf_leading run start={lead_run_start} end={end} len={length}", flush=True)
        if osc:
            osc.send_message("/event", ["perf_leading", lead_run_start, end, length])
    if jump_run_start is not None:
        end = len(df) - 1
        length = end - jump_run_start + 1
        print(f"[event] score_jump run start={jump_run_start} end={end} len={length}", flush=True)
        if osc:
            osc.send_message("/event", ["score_jump", jump_run_start, end, length])

    summarize_runs(df)
    return df




#keeps throwing errors when attempting to start stream on perf_idx==0. Resorting to flushing out 
# most of the play audio mechanism for the sake of livetime streaming accuracy. 

#started by defining pyAudio, defining a bigger buffer, 
def _start_playback(performance_path: Path, stop_event: threading.Event) -> threading.Thread:
    audio, sr = librosa.load(performance_path, sr=None)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio.astype(np.float32)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
    )
    #audio buffer construction
    chunk_size = 2048

    def _playback_worker():
        try:
            idx = 0
            while idx < len(audio) and not stop_event.is_set():
                chunk = audio[idx : idx + chunk_size]
                stream.write(chunk.tobytes())
                idx += chunk_size
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    thread = threading.Thread(
        target=_playback_worker,
        name="matchmaker-audio-playback",
        daemon=True,
    )
    thread.start()
    return thread


# def describe_tempo_events(df: pd.DataFrame) -> List[dict]:
#     # Placeholder: not currently used in the CLI
#     # Returns an empty list to keep 
#     return [] 






#the main matchmaker helper, initializes everything
def stream_live_run(
    mm: Matchmaker,
    frame_rate: int,
    run_name: str,
    save_path: Optional[Path], #for newly annotated paths
    annotations: Optional[Path], #!!!!
    results_dir: Path,
    verbose_run: bool,
    playback: bool,
    osc: Optional["SimpleUDPClient"] = None,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    rows: List[dict] = []
    prev_ref = None
    prev_perf = None
    playback_thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    playback_started = False
    lead_run_start: Optional[int] = None
    jump_run_start: Optional[int] = None

    #start stream alignment, flushing of playback mechanism onto my own platform. 
    for frame_idx, _ in enumerate(mm.run(verbose=verbose_run), start=0):
        if (
            playback
            and not playback_started
            and mm.input_type == "audio"
            and mm.performance_file is not None
            and Path(mm.performance_file).exists()
        ):
            stop_event = threading.Event()
            playback_thread = _start_playback(
                performance_path=Path(mm.performance_file),
                stop_event=stop_event,
            )
            playback_started = True

        ref_idx, perf_idx = mm.score_follower._warping_path[-1]
        delta_ref = ref_idx - prev_ref if prev_ref is not None else np.nan
        delta_perf = perf_idx - prev_perf if prev_perf is not None else np.nan
        dt_score = delta_ref / frame_rate if prev_ref is not None else np.nan
        dt_perf = delta_perf / frame_rate if prev_perf is not None else np.nan
        dt_score_val = dt_score
        delta_ratio = np.nan
        if not np.isnan(delta_ref) and delta_ref != 0:
            delta_ratio = delta_perf / delta_ref
        elif not np.isnan(delta_perf):
            delta_ratio = delta_perf  # capture perf advance when score is clamped
        tempo_ratio = delta_ratio if (not np.isnan(delta_ref) and delta_ref != 0) else np.nan
        perf_leading = (delta_ref == 0 or np.isnan(delta_ref)) and (delta_perf is not None) and not np.isnan(delta_perf) and delta_perf > 0

        #more debugging for performance
        print(
            f"Δref={delta_ref if not np.isnan(delta_ref) else 'nan'}, Δperf={delta_perf if not np.isnan(delta_perf) else 'nan'}, "
            f"grad={_format_ratio(delta_ratio)}"
        )

        if osc:
            osc.send_message("/warp", [
                int(ref_idx),
                int(perf_idx),
                float(delta_ratio),
                float(tempo_ratio) if not np.isnan(tempo_ratio) else np.nan,
                float(dt_score_val),
                float(delta_ref),
                float(delta_perf),
                int(perf_leading),
            ])

        # store per-frame score delta for debugging/analysis
        prev_ref = ref_idx
        prev_perf = perf_idx

        rows.append(
            {
                "frame": frame_idx,
                "ref_idx": ref_idx,
                "input_idx": perf_idx,
                "score_time": ref_idx / frame_rate,
                "perf_time": perf_idx / frame_rate,
                "dt_score": dt_score_val,
                "delta_ref": delta_ref,
                "delta_perf": delta_perf,
                "delta_ratio": delta_ratio,
                "tempo_ratio": tempo_ratio,
                "perf_leading": perf_leading,
            }
        )
        tr_display = _format_ratio(tempo_ratio if not np.isnan(tempo_ratio) else delta_ratio)
        print(
            f"[live] Frame {frame_idx:05d} | score_idx={ref_idx} | "
            f"perf_idx={perf_idx} | t_r={_format_ratio(tempo_ratio)} | "
            f"grad={_format_ratio(delta_ratio)} | dt_score={_format_ratio(dt_score_val)}",
            flush=True,
        )

        # event logging as soon as runs close
        lead_now = (delta_ref == 0 or np.isnan(delta_ref)) and (not np.isnan(delta_perf)) and delta_perf > 0
        jump_now = (not np.isnan(delta_ref)) and delta_ref >= 5
        if lead_now and lead_run_start is None:
            lead_run_start = frame_idx
        if not lead_now and lead_run_start is not None:
            end = frame_idx - 1
            length = end - lead_run_start + 1
            print(f"[event] perf_leading run start={lead_run_start} end={end} len={length}", flush=True)
            if osc:
                osc.send_message("/event", ["perf_leading", lead_run_start, end, length])
            lead_run_start = None

        if jump_now and jump_run_start is None:
            jump_run_start = frame_idx
        if not jump_now and jump_run_start is not None:
            end = frame_idx - 1
            length = end - jump_run_start + 1
            print(f"[event] score_jump run start={jump_run_start} end={end} len={length}", flush=True)
            if osc:
                osc.send_message("/event", ["score_jump", jump_run_start, end, length])
            jump_run_start = None

    if playback_thread is not None:
        # Let playback finish naturally for live runs too
        playback_thread.join()

    # flush any open runs
    if lead_run_start is not None:
        end = frame_idx
        length = end - lead_run_start + 1
        print(f"[event] perf_leading run start={lead_run_start} end={end} len={length}", flush=True)
        if osc:
            osc.send_message("/event", ["perf_leading", lead_run_start, end, length])
    if jump_run_start is not None:
        end = frame_idx
        length = end - jump_run_start + 1
        print(f"[event] score_jump run start={jump_run_start} end={end} len={length}", flush=True)
        if osc:
            osc.send_message("/event", ["score_jump", jump_run_start, end, length])

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Matchmaker run produced no frames; check input files.")
    print(
        "[live] Collected "
        f"{len(df)} frames (max ref_idx={int(df['ref_idx'].max())}, "
        f"max input_idx={int(df['input_idx'].max())})",
        flush=True,
    )
    summarize_runs(df)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, sep="\t", index=False)
#haven't ran a live test yet.... 
    eval_results = None
    if annotations:
        if mm.performance_file is None:
            print("[live] Skipping evaluation/debug save because no performance file was provided.", flush=True)
        else:
            eval_results = mm.run_evaluation(
                perf_annotations=annotations,
                debug=True,
                save_dir=results_dir,
                run_name=run_name,
            )
            print(f"[live] Evaluation summary:\n{json.dumps(eval_results, indent=2)}")

    return df, eval_results



# shifting to allow analyze_warp to be the main module. Taken from stack overflow. 

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Matchmaker warping paths by streaming tempo ratios "
            "frame-by-frame either from a saved TSV or from a fresh Matchmaker run."
        )
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "wp_simple_example.tsv",
        help="Warp-path TSV produced by run_examples/run_evaluation.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=FRAME_RATE,
        help="Frame rate used when generating warp path.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run Matchmaker directly and print tempo ratios as frames arrive.",
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        default=DEFAULT_SCORE,
        help="Score file to use when --live is set.",
    )
    parser.add_argument(
        "--performance-file",
        type=Path,
        default=DEFAULT_AUDIO_PERF,
        help="audio or MIDI performance file, only when  --live is set. "
        "Pass an empty string to use live input instead of a file.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=(
            "Annotation file for evaluation when --live is set. "
            "Defaults to matchmaker/assets/simple_perf_annotations.txt when available."
        ),
    )
    parser.add_argument(
        "--input-type",
        choices=["audio", "midi"],
        default="audio",
        help="Input modality for the Matchmaker live run.",
    )
    parser.add_argument(
        "--method",
        choices=["arzt", "dixon", "hmm", "pthmm"],
        default=None,
        help="Score-following backend to use. Defaults to Matchmaker's per-input default.",
    )
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        help="Respect real-time pacing during live runs (default).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Run live alignment as fast as possible (disables real-time pacing).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "wp_simple_example_with_tempo.tsv",
        help="Optional path to store the enriched warp path.",
    )
    parser.add_argument(
        "--playback",
        dest="playback",
        action="store_true",
        help="Play the performance audio while streaming live frames (default).",
    )
    parser.add_argument(
        "--no-playback",
        dest="playback",
        action="store_false",
        help="Disable performance audio playback during live runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory used for Matchmaker debug/evaluation artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="simple_example",
        help="Label used for saved artifacts when --live is set.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable Matchmaker's internal progress bars/logs.",
    )
    #TouchDesigner flags
    parser.add_argument(
        "--osc-host", default="127.0.0.1"
    )
    parser.add_argument(
        "--osc-port", type=int, default=8000
    )
    parser.add_argument(
        "--osc", action="store_true", 
        help="Send OSC to TouchDesigner application– Clarify host and port in order to work"
    )
    
    parser.set_defaults(wait=True, playback=True)
    return parser.parse_args(argv)


def _reset_audio_hop(mm: Matchmaker, frame_rate: int) -> None:
    """Rebuild audio pipeline to honor a custom frame rate without touching core code."""
    hop_length = max(1, int(round(SAMPLE_RATE / frame_rate)))
    proc_cls = type(mm.processor)
    if proc_cls not in {
        ChromagramProcessor,
        MFCCProcessor,
        CQTProcessor,
        MelSpectrogramProcessor,
        LogSpectralEnergyProcessor,
    }:
        print("[warn] Custom frame_rate ignored: unsupported processor type for audio hop override.")
        return

    # Recreate processor and stream with new hop
    new_proc = proc_cls(sample_rate=SAMPLE_RATE, hop_length=hop_length)
    prev_stream = mm.stream
    performance_file = Path(mm.performance_file) if mm.performance_file else None
    new_stream = AudioStream(
        processor=new_proc,
        file_path=str(performance_file) if performance_file else None,
        wait=getattr(prev_stream, "wait", True),
        target_sr=SAMPLE_RATE,
        hop_length=hop_length,
        sample_rate=SAMPLE_RATE,
        queue=getattr(prev_stream, "queue", None),
        device_name_or_index=getattr(mm, "device_name_or_index", None),
    )

    # Recompute reference features at the new hop
    mm.processor = new_proc
    mm.stream = new_stream
    mm.frame_rate = frame_rate
    mm.reference_features = mm.processor(mm.score_audio)

    # Recreate score follower with the new reference and queue
    sf = mm.score_follower
    if isinstance(sf, OnlineTimeWarpingArzt):
        dist = getattr(sf, "distance_func", OnlineTimeWarpingArzt.DEFAULT_DISTANCE_FUNC)
        mm.score_follower = OnlineTimeWarpingArzt(
            reference_features=mm.reference_features,
            queue=mm.stream.queue,
            distance_func=dist,
            frame_rate=frame_rate,
        )
    elif isinstance(sf, OnlineTimeWarpingDixon):
        dist = getattr(sf, "distance_func", OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC)
        mm.score_follower = OnlineTimeWarpingDixon(
            reference_features=mm.reference_features,
            queue=mm.stream.queue,
            distance_func=dist,
            frame_rate=frame_rate,
        )
    elif isinstance(sf, GaussianAudioPitchTempoHMM):
        mm.score_follower = GaussianAudioPitchTempoHMM(
            reference_features=mm.reference_features,
            queue=mm.stream.queue,
            transition_scale=getattr(sf, "transition_scale", 0.05),
        )
    elif isinstance(sf, GaussianAudioPitchHMM):
        mm.score_follower = GaussianAudioPitchHMM(
            reference_features=mm.reference_features,
            queue=mm.stream.queue,
        )
    else:
        print("[warn] Custom frame_rate applied to features/stream, but score_follower type not handled.")


def main() -> None:
    args = parse_args()
    #create TouchDesginer client
    osc = SimpleUDPClient(args.osc_host, args.osc_port) if args.osc else None


    if not args.live:
        df = stream_from_file(
            tsv_path=args.tsv,
            frame_rate=args.frame_rate,
            playback=args.playback,
            performance_path=args.performance_file,
            wait=args.wait,
            osc=osc,
        )
        if args.save_path:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_path, sep="\t", index=False)
        return

    annotations = args.annotations
    if annotations is None and DEFAULT_ANNOTS.exists():
        annotations = DEFAULT_ANNOTS

    performance_file: Optional[Path] = (
        args.performance_file
        if args.performance_file
        and str(args.performance_file) not in ("", ".")
        else None
    )

    mm = Matchmaker(
        score_file=args.score_file,
        performance_file=performance_file,
        wait=args.wait,
        input_type=args.input_type,
        method=args.method,
        frame_rate=args.frame_rate,
    )
    # Optional audio hop override using CLI frame_rate without touching core Matchmaker code
    if args.input_type == "audio":
        _reset_audio_hop(mm, args.frame_rate)
    #element that starts audio upon loop init.... not reliable. 
    #resolved^
    stream_live_run(
        mm=mm,
        frame_rate=args.frame_rate,
        run_name=args.run_name,
        save_path=args.save_path,
        annotations=annotations,
        results_dir=args.results_dir,
        verbose_run=not args.quiet,
        playback=args.playback,
        osc=osc,
    )


if __name__ == "__main__":
    main()
