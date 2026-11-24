"""Render a 3/4/3/4-second debate using one continuous streaming inference pass."""
import csv

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT, SPEAKERS
from male_study import ROOT, read_case
from src.audio_io import write_audio
from src.calibration import load_profiles
from src.debate_controller import Controller, RATE, Router
from src.evaluation import projection_gains
from src.streaming import StreamingSeparator


def render(directory, replay):
    profiles,_ = load_profiles(ROOT/"comparisons"/"profiles"/"development_640.json")
    x, refs = read_case("development",0)
    # Repeat both aligned references together to cover two whole rounds.
    n = 14*RATE
    refs = np.tile(refs,(1,2))[:,:n]
    x = np.tile(x,2)[:n]
    y = replay(StreamingSeparator(profiles),x).T
    ctl = Controller(lambda:0)
    ctl.command("start",(3,4,"male",True,2))
    router = Router(ctl)
    blocks, events = [], []
    for pos in range(0,n,197):
        block, changes = router.apply(pos,y[pos:pos+197],x[pos:pos+197])
        blocks.append(block)
        events.extend(changes)
    selected = np.concatenate(blocks)
    assert [e[0] for e in events] == [0,3*RATE,7*RATE,10*RATE]
    assert ctl.state(n).state == "STOPPED"
    folder = directory/"debate"
    for name,data in zip(("mixture","selected",*SPEAKERS),(x,selected,*y.T)):
        write_audio(folder/f"{name}.wav",data,DEFAULT.rate)
    with (folder/"turns.csv").open("w",newline="") as f:
        w = csv.writer(f)
        w.writerow(("sample","seconds","state","route","round","target_gain_db","interferer_gain_db"))
        for j,(t,state,route,turn) in enumerate(events):
            k = SPEAKERS.index(route)
            end = events[j+1][0] if j+1<len(events) else n
            region = slice(t+router.fade,end)
            np.testing.assert_allclose(selected[region],y[region,k],atol=1e-6)
            gains = projection_gains(selected[region],refs[:,region])
            assert gains[k] > gains[1-k], (route,gains)
            w.writerow((t,t/RATE,state,route,1+turn//2,gains[k],gains[1-k]))
        w.writerow((n,n/RATE,"STOPPED","mute",2,None,None))
    fig,(ax,zoom) = plt.subplots(2,1,figsize=(10,5),layout="constrained")
    timeline = np.arange(n)/RATE
    ax.plot(timeline[::16],selected[::16],lw=.5)
    for t,_,route,_ in events:
        ax.axvline(t/RATE,color="gray",lw=1)
        ax.text(t/RATE+.07,.90,route,transform=ax.get_xaxis_transform())
    ax.set(xlabel="Capture time (s)",ylabel="Selected amplitude")
    region = slice(3*RATE-160,3*RATE+640)
    for k,name in enumerate(SPEAKERS):
        zoom.plot(timeline[region],y[region,k],label=name,alpha=.6)
    zoom.plot(timeline[region],selected[region],label="selected",lw=1)
    zoom.axvspan(3,3+.02,color="grey",alpha=.15)
    zoom.set(xlabel="First transition (s)",ylabel="Amplitude")
    zoom.legend()
    fig.savefig(folder/"transitions.png",dpi=120)
    plt.close(fig)
