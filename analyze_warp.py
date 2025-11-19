import argparse
import json
import os
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import pyaudio

from matchmaker import Matchmaker
from matchmaker.features.audio import SAMPLE_RATE


os.environ["PARTITURA_SOUNDFONT"] = "tests/Steinway_B_soundfont.sf2"

DEFAULT_SCORE = Path("matchmaker/assets/simple_score.musicxml")

#An unclean performance
DEFAULT_AUDIO_PERF = Path("matchmaker/assets/simple_performance.mp3")
DEFAULT_ANNOTS = Path("matchmaker/assets/simple_perf_annotations.txt")
DEFAULT_RESULTS_DIR = Path("results")

FRAME_RATE = 60


def _add_temporal_columns(df: pd.DataFrame, frame_rate: int) -> pd.DataFrame:
    df["score_time"] = df["ref_idx"] / frame_rate
    df["perf_time"] = df["input_idx"] / frame_rate


#append to NaN instead of 0 to prevent
    dt_score = np.diff(df["score_time"].to_numpy(), prepend=np.nan)
    dt_perf = np.diff(df["perf_time"].to_numpy(), prepend=np.nan)

    tempo_ratio = np.divide(
        dt_perf,
        dt_score,
        out=np.full_like(dt_perf, np.nan),
        where=np.abs(dt_score) > 0,
    )
    df["tempo_ratio"] = tempo_ratio
    return df


#account for nan
def _format_ratio(ratio: float) -> str:
    if ratio is None or np.isnan(ratio):
        return "nan"
    return f"{ratio:.4f}"


# run matchmakers  ref vs. input
def stream_from_file(tsv_path: Path, frame_rate: int) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["ref_idx", "input_idx"],
    )
    df = _add_temporal_columns(df, frame_rate)

    for idx, row in df.iterrows():
        ratio_str = _format_ratio(row["tempo_ratio"])
        print(
            f"[file] Frame {idx:05d} | score_idx={int(row['ref_idx'])} | "
            f"perf_idx={int(row['input_idx'])} | tempo_ratio={ratio_str}",
            flush=True,
        )

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



#another run-matchmaker helper, initializes state. 
def stream_live_run(
    mm: Matchmaker,
    frame_rate: int,
    run_name: str,
    save_path: Optional[Path], #for newly annotated paths
    annotations: Optional[Path], #!!!!
    results_dir: Path,
    verbose_run: bool,
    playback: bool,
) -> Tuple[pd.DataFrame, Optional[dict]]:
    rows: List[dict] = []
    prev_ref = None
    prev_perf = None
    playback_thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    playback_started = False

    #start stream alignment, more flushing of playback onto my own platform. 
    try:
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
            dt_score = (
                (ref_idx - prev_ref) / frame_rate if prev_ref is not None else np.nan
            )
            dt_perf = (
                (perf_idx - prev_perf) / frame_rate if prev_perf is not None else np.nan
            )
            tempo_ratio = (
                dt_perf / dt_score
                if dt_score not in (None, 0) and not np.isnan(dt_score)
                else np.nan
            )
            prev_ref = ref_idx
            prev_perf = perf_idx

            rows.append(
                {
                    "frame": frame_idx,
                    "ref_idx": ref_idx,
                    "input_idx": perf_idx,
                    "score_time": ref_idx / frame_rate,
                    "perf_time": perf_idx / frame_rate,
                    "tempo_ratio": tempo_ratio,
                }
            )
            print(
                f"[live] Frame {frame_idx:05d} | score_idx={ref_idx} | "
                f"perf_idx={perf_idx} | tempo_ratio={_format_ratio(tempo_ratio)}",
                flush=True,
            )
    finally:
        if stop_event is not None:
            stop_event.set()
        if playback_thread is not None:
            playback_thread.join(timeout=1)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Matchmaker run produced no frames; check input files.")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, sep="\t", index=False)
#haven't ran a live test yet.... 
    eval_results = None
    if annotations:
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
        help="audio or MIDI performance file, only when  --live is set.", #
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
    parser.set_defaults(wait=True, playback=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.live:
        df = stream_from_file(args.tsv, args.frame_rate)
        if args.save_path:
            args.save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_path, sep="\t", index=False)
        return

    annotations = args.annotations
    if annotations is None and DEFAULT_ANNOTS.exists():
        annotations = DEFAULT_ANNOTS

    mm = Matchmaker(
        score_file=args.score_file,
        performance_file=args.performance_file,
        wait=args.wait,
        input_type=args.input_type,
        method=args.method,
        frame_rate=args.frame_rate,
    )
    #element that starts audio.... but there are still some
    stream_live_run(
        mm=mm,
        frame_rate=args.frame_rate,
        run_name=args.run_name,
        save_path=args.save_path,
        annotations=annotations,
        results_dir=args.results_dir,
        verbose_run=not args.quiet,
        playback=args.playback,
    )


if __name__ == "__main__":
    main()
