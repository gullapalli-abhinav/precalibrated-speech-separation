"""Exercise the live mic/duplex worker at zero playback gain, with aggregate telemetry only."""
import argparse
from time import monotonic, process_time, sleep

import numpy as np
import psutil

from config import OUTPUT
from src.calibration import write_json
from src.debate_controller import Controller
from src.enrollment import read_session
from src.live_audio import LiveAudio, devices, rates


def check(profiles, inp, out, seconds):
    ctl = Controller()
    live = LiveAudio(profiles,ctl)
    live.volume = 0
    ctl.command("start",(3,4,"male",True,0))
    live.start(inp,out)
    process = psutil.Process()
    start,cpu = monotonic(),process_time()
    latencies, rss = [], []
    try:
        while monotonic()-start < seconds:
            sleep(.25)
            if live.error or not live.stream.active:
                raise RuntimeError(live.error or "Device disconnected")
            if live.latency_ms is not None:
                latencies.append(live.latency_ms)
            rss.append(process.memory_info().rss/2**20)
        elapsed = monotonic()-start
        hardware = live.stream.latency
    finally:
        live.close()
    t = np.asarray(live.engine.times)*1000
    return dict(seconds=elapsed,rate=live.rate,frames=live.engine.frame_count,
        median_p95_p99_max_frame_ms=np.percentile(t,[50,95,99,100]).tolist(),misses=live.engine.misses,
        underflows=live.underflows,overflows=live.overflows,peak_input_samples=live.input.peak,
        peak_output_samples=live.output.peak,declared_io_latency_ms=(1000*np.array(hardware)).tolist(),
        estimated_adc_dac_p50_p95_max_ms=np.percentile(latencies,[50,95,100]).tolist() if latencies else None,
        rss_first_last_max_mib=[rss[0],rss[-1],max(rss)],cpu_core_percent=100*(process_time()-cpu)/elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices",action="store_true")
    parser.add_argument("--input",type=int)
    parser.add_argument("--output",type=int)
    parser.add_argument("--profiles")
    parser.add_argument("--seconds",type=float,default=60)
    args = parser.parse_args()
    if args.devices:
        for i,d in devices():
            print(i,d["name"],"inputs",d["max_input_channels"],"outputs",d["max_output_channels"],"host",d["hostapi"])
    else:
        if None in (args.input,args.output,args.profiles) or args.seconds <= 0:
            parser.error("Supply input and output device IDs, local profiles and a positive duration")
        p = read_session(args.profiles)
        results = dict(compatible_rates=list(rates(args.input,args.output)),runs=[])
        for duration in (4,args.seconds):
            result = check(p,args.input,args.output,duration)
            print(result,flush=True)
            results["runs"].append(result)
        write_json(OUTPUT/"realtime"/"hardware_check.json",results)
