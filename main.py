"""Offline selective muting CLI."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from config import DEFAULT, DEFAULT_METHOD, FEMALE, LEGACY_DIGITAL, MALE, METHODS, MODES, OUTPUT, REAL, SPEAKERS
from src.audio_io import read_audio
from src.calibration import create_profiles, load_profiles, write_json
from src.evaluation import evaluate, save_tracks, summary
from src.separator import MALE_VARIANTS, analyze, save_outputs, separate


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="show WAV format, levels and clipping indicators")
    inspect.add_argument("files", nargs="*", type=Path, default=[FEMALE, MALE, REAL, LEGACY_DIGITAL])

    calibrate = sub.add_parser("calibrate", help="learn fixed profiles from calibration ranges")
    calibrate.add_argument("--cepstral", action="store_true", help="also save 2 ms envelope examples for the optional male mask")
    calibrate.add_argument("--window-ms", type=int, choices=(32, 40, 64), default=40)
    evaluation = sub.add_parser("evaluate", help="generate and score held-out mixtures, and process the real recording")
    for command in (calibrate, evaluation):
        command.add_argument("--female", type=Path, default=FEMALE)
        command.add_argument("--male", type=Path, default=MALE)
    evaluation.add_argument("--input", type=Path, default=REAL, help="real shared-microphone WAV")
    evaluation.add_argument("--supplied-digital", type=Path, help="optional legacy digital WAV without aligned references")
    evaluation.add_argument("--skip-supplied-digital", action="store_true")

    processing = (
        sub.add_parser("separate", help="produce speaker, mixture and selected output WAVs"),
        sub.add_parser("visualize", help="plot mixture, pitch, masks and output spectrograms"),
    )
    for command in processing:
        command.add_argument("--input", type=Path, default=REAL)
        command.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
        command.add_argument("--male-variant", choices=MALE_VARIANTS, default="current")
        command.add_argument("--male-floor", type=float, default=.22, help="maximum relaxation for conservative_mask")
        command.add_argument("--power-exponent", type=float, default=.75, help="exponent for power_ratio")
    processing[0].add_argument("--mode", choices=MODES, default="both")
    processing[0].add_argument("--male-window-profiles", type=Path,
                               help="use a second calibrated analysis window for the male output")
    for command in (calibrate, evaluation, *processing):
        command.add_argument("--out-dir", type=Path, default=OUTPUT)
        command.add_argument("--profiles", type=Path, help="defaults to profiles.json in the output directory")
    return cli


def run(args: argparse.Namespace) -> None:
    if args.command == "inspect":
        for path in args.files:
            _, info = read_audio(path)
            print(f"{path.name:30} {info['rate_hz']:5} Hz  {info['channels']} ch  "
                  f"{info['decoded_dtype']:7}  {info['duration_seconds']:7.2f} s  "
                  f"peak {info['peak']:.5f}  RMS {info['rms']:.5f}  rails {info['rail_samples']}")
        return
    profile_path = args.profiles or args.out_dir / "profiles.json"
    if args.command == "calibrate":
        from src.visualization import calibration_plots

        cfg = replace(DEFAULT, window=args.window_ms * DEFAULT.rate // 1000)
        profiles, diagnostics = create_profiles((args.female, args.male), profile_path, cfg,
                                                include_cepstral=args.cepstral)
        calibration_plots(args.out_dir / "calibration", profiles, diagnostics, cfg)
        for name in SPEAKERS:
            profile = profiles[name]
            print(f"{name}: median F0 {profile['pitch']['median_hz']:.1f} Hz, {profile['voiced_frames']} voiced frames")
        print(f"Saved profiles to {profile_path}")
    elif args.command == "evaluate":
        supplied = None if args.skip_supplied_digital else (
            args.supplied_digital or (LEGACY_DIGITAL if LEGACY_DIGITAL.is_file() else None)
        )
        evaluate((args.female, args.male), args.input, supplied, profile_path, args.out_dir)
        print(f"Saved results to {args.out_dir / 'evaluation.json'}")
    else:
        profiles, cfg = load_profiles(profile_path)
        x, info = read_audio(args.input, cfg.rate)
        analysis = analyze(x, profiles, cfg)
        alternate = getattr(args, "male_window_profiles", None)
        result = separate(analysis, len(x), profiles, cfg, args.method, male_variant="current" if alternate else args.male_variant,
                          male_floor=args.male_floor, power_exponent=args.power_exponent)
        if alternate:
            male_profiles, male_cfg = load_profiles(alternate)
            if (male_cfg.rate, male_cfg.hop, male_cfg.fft) != (cfg.rate, cfg.hop, cfg.fft):
                raise ValueError("Male-window profiles must share the sample rate, hop and FFT grid")
            for name in SPEAKERS:
                for key in ("sha256", "calibration_samples"):
                    if male_profiles[name]["recording"][key] != profiles[name]["recording"][key]:
                        raise ValueError("Male-window profiles must use the same speaker calibration ranges")
            male_analysis = analyze(x, male_profiles, male_cfg)
            male_result = separate(male_analysis, len(x), male_profiles, male_cfg, args.method,
                                   male_variant=args.male_variant, male_floor=args.male_floor,
                                   power_exponent=args.power_exponent)
            result.audio[1] = male_result.audio[1]
            result.masks[:, 1] = male_result.masks[:, 1]
            analysis.tracks.f0[:, 1] = male_analysis.tracks.f0[:, 1]
            analysis.tracks.confidence[:, 1] = male_analysis.tracks.confidence[:, 1]
            analysis.seconds += male_analysis.seconds
            result.seconds += male_result.seconds
        if args.command == "separate":
            save_outputs(args.out_dir, x, result, cfg, args.mode)
            save_tracks(args.out_dir / "pitch.csv", analysis, cfg)
            if alternate:
                save_tracks(args.out_dir / "pitch_male_window.csv", male_analysis, male_cfg)
            write_json(args.out_dir / "diagnostics.json", {
                "recording": info, "method": args.method, "male_variant": args.male_variant,
                "male_floor": args.male_floor, "power_exponent": args.power_exponent,
                "male_window_samples": male_cfg.window if alternate else cfg.window,
                **summary(x, analysis, result, cfg),
            })
        else:
            from src.visualization import separation_plots

            separation_plots(args.out_dir, x, analysis, result, cfg, f"{args.input.name}, {args.method}")
        print(f"Saved results to {args.out_dir}")


def main() -> int:
    cli = parser()
    try:
        run(cli.parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
