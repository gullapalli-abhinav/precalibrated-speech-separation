"""Launch the local Tk debate console with python realtime_app.py."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np

from config import OUTPUT, SPEAKERS
from src.audio_io import read_audio
from src.calibration import write_json
from src.debate_controller import Controller, RATE, TURNS, now_tick
from src.enrollment import enroll, read_session, session_profiles
from src.live_audio import LiveAudio, backend, devices, rates, record_sample


class App:
    def __init__(self, root):
        self.root = root
        root.title("Pre-calibrated speech separation")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = self.live = None
        self.cancel = Event()
        self.closing = False
        self.controller = Controller()
        self.profiles = {}
        self.notice = "Choose devices and enroll both speakers."
        self.failed = False
        self.progress = None
        self.input, self.output = tk.StringVar(), tk.StringVar()
        self.duration = tk.StringVar(value="30")
        self.first = tk.StringVar(value="male")
        self.auto, self.record = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.volume = tk.DoubleVar(value=.15)
        self.turn = {s:tk.StringVar(value="45") for s in SPEAKERS}
        self.rounds, self.fade = tk.StringVar(value="0"), tk.StringVar(value="20")
        self.status = tk.StringVar()
        self.profile_status = tk.StringVar()

        audio, cal, debate = [ttk.LabelFrame(root,text=title,padding=8) for title in
                              ("Audio configuration","Speaker calibration","Debate")]
        for frame in (audio,cal,debate):
            frame.pack(fill="x",padx=8,pady=4)
        self.inputs = self.field(audio,0,"Input microphone",self.input,(),width=54)
        self.outputs = self.field(audio,1,"Output device",self.output,(),width=54)
        actions = ttk.Frame(audio)
        actions.grid(row=2,columnspan=2,sticky="w")
        self.button(actions,"Refresh",self.refresh)
        self.button(actions,"Test input + quiet tone",self.audio_test)
        ttk.Label(actions,text="Output volume").pack(side="left")
        ttk.Scale(actions,from_=0,to=1,variable=self.volume,length=120).pack(side="left")
        self.meters = []
        for row, name in enumerate(("Input level","Output level"),3):
            ttk.Label(audio,text=name).grid(row=row,column=0,sticky="w")
            bar = ttk.Progressbar(audio,maximum=1,length=300)
            bar.grid(row=row,column=1,sticky="ew")
            self.meters.append(bar)
        ttk.Label(audio,text="Use headphones. Disable microphone monitoring and OS loopback.").grid(row=5,columnspan=2)
        self.field(cal,0,"Record duration (seconds)",self.duration,("30","45","60","90"))
        ttk.Label(cal,text="Only the speaker being enrolled should speak.").grid(row=1,columnspan=2,sticky="w")
        for row,s in enumerate(("male","female"),2):
            group = ttk.Frame(cal)
            group.grid(row=row,columnspan=2,sticky="w")
            self.button(group,f"Record / recalibrate {s}",lambda s=s:self.calibrate(s,True))
            self.button(group,f"Import {s} WAV",lambda s=s:self.calibrate(s,False))
        group = ttk.Frame(cal)
        group.grid(row=4,columnspan=2,sticky="w")
        self.button(group,"Save profiles",self.save)
        self.button(group,"Load profiles",self.load)
        ttk.Label(cal,textvariable=self.profile_status).grid(row=5,columnspan=2,sticky="w")
        for row,s in enumerate(("male","female")):
            self.field(debate,row,s.capitalize()+" seconds",self.turn[s])
        self.field(debate,2,"Starting speaker",self.first,("male","female"))
        self.field(debate,3,"Rounds (0 = continuous)",self.rounds)
        self.field(debate,4,"Crossfade (ms)",self.fade)
        options = ttk.Frame(debate)
        options.grid(row=5,columnspan=2,sticky="w")
        ttk.Checkbutton(options,text="Automatic turns",variable=self.auto,
                        command=lambda:self.controller.command("auto",self.auto.get())).pack(side="left")
        ttk.Checkbutton(options,text="Record next session locally",variable=self.record).pack(side="left")
        for row,buttons in enumerate((
            (("Start",self.start),("Pause / resume",self.pause),("Next speaker",lambda:self.controller.command("next")),("Stop",self.stop)),
            tuple((s.capitalize(),lambda s=s:self.select(s)) for s in ("male","female","both","mute"))),6):
            group = ttk.Frame(debate)
            group.grid(row=row,columnspan=2,sticky="w")
            for title,fn in buttons:
                self.button(group,title,fn)
        self.label = tk.Label(root,textvariable=self.status,anchor="w",justify="left",wraplength=620)
        self.label.pack(fill="x",padx=12,pady=6)
        root.protocol("WM_DELETE_WINDOW",self.quit)
        self.refresh()
        self.poll()

    def field(self,frame,row,label,var,choices=None,width=12):
        ttk.Label(frame,text=label).grid(row=row,column=0,sticky="w",padx=3)
        if choices is None:
            widget = ttk.Entry(frame,textvariable=var,width=width)
        else:
            widget = ttk.Combobox(frame,textvariable=var,values=choices,state="readonly",width=width)
        widget.grid(row=row,column=1,sticky="w",pady=1)
        return widget

    def button(self,parent,label,fn):
        def click():
            try:
                fn()
            except Exception as exc:
                self.notice,self.failed = str(exc),True
        ttk.Button(parent,text=label,command=click).pack(side="left",padx=2,pady=2)

    def idle(self):
        if self.live or self.job:
            raise ValueError("Stop the current session or wait for the current operation")

    def task(self,notice,work,success):
        self.notice,self.failed = notice,False
        self.job = self.pool.submit(work),success

    def refresh(self):
        self.idle()
        def loaded(items):
            for widget,var,key in ((self.inputs,self.input,"max_input_channels"),
                                   (self.outputs,self.output,"max_output_channels")):
                opts = [f"{i}  {d['name']}" for i,d in items if d[key]]
                widget["values"] = opts
                if var.get() not in opts:
                    var.set(opts[0] if opts else "")
            self.notice = "Devices refreshed. Use input and output on the same host API."
        self.task("Enumerating audio devices…",devices,loaded)

    def device(self,var):
        if not var.get():
            raise ValueError("Select an available device first")
        return int(var.get().split()[0])

    def audio_test(self):
        self.idle()
        inp,out = self.device(self.input),self.device(self.output)
        gain = min(.04,self.volume.get())
        def test():
            sd = backend()
            for rate in rates(inp,out):
                try:
                    stream = sd.Stream(device=(inp,out),channels=1,samplerate=rate,dtype="float32")
                except sd.PortAudioError:
                    continue
                break
            else:
                raise ValueError("No compatible duplex device pair")
            peak = 0.
            with stream:
                t = np.arange(rate//100)/rate
                for _ in range(50):
                    if self.cancel.is_set():
                        break
                    stream.write((gain*np.sin(2*np.pi*400*t)).astype('float32'))
                    data,_ = stream.read(len(t))
                    peak = max(peak,float(abs(data).max()))
            return f"Duplex test completed at {rate} Hz. Input peak {peak:.3f}."
        self.cancel.clear()
        self.task("Testing devices…",test,self.message)

    def message(self,msg):
        self.notice = msg

    def ready(self):
        self.controller.command("READY" if len(self.profiles)==2 else "IDLE")

    def calibrate(self,s,record):
        self.idle()
        if record:
            device = self.device(self.input)
            seconds = int(self.duration.get())
        else:
            filename = filedialog.askopenfilename(filetypes=[("WAV recordings","*.wav")])
            if not filename:
                return
        self.controller.command("CALIBRATING_"+s.upper())
        self.cancel.clear()
        def work():
            if record:
                x,clipped = record_sample(device,seconds,self.level,self.cancel)
                return enroll(x,expected_seconds=seconds,clipped_fraction=clipped)
            x,info = read_audio(Path(filename))
            fraction = info["rail_samples"]/(info["duration_seconds"]*info["rate_hz"]*info["channels"])
            return enroll(x,clipped_fraction=fraction)
        def done(profile):
            self.profiles[s] = profile
            self.notice = f"{s.capitalize()} enrolled."
            self.ready()
        self.task(f"Calibrating {s}…",work,done)

    def level(self,remaining,peak):
        self.progress = remaining,peak

    def load(self):
        self.idle()
        path = filedialog.askopenfilename(filetypes=[("Local profiles","*.json")])
        if path:
            def loaded(profiles):
                self.profiles = profiles
                self.ready()
                self.notice = "Both profiles loaded."
            self.task("Loading profiles…",lambda:read_session(Path(path)),loaded)

    def save(self):
        self.idle()
        profiles = session_profiles(self.profiles) if len(self.profiles)==2 else None
        if profiles is None:
            raise ValueError("Enroll both speakers before saving")
        folder = OUTPUT/"realtime"
        folder.mkdir(parents=True,exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=folder,initialfile="debate.profile.json",
                                             defaultextension=".profile.json",filetypes=[("Profiles","*.profile.json")])
        if path:
            if "profile" not in Path(path).name.lower() or not path.lower().endswith(".json"):
                raise ValueError("Use a filename ending in .profile.json for Git exclusion")
            self.task("Saving locally…",lambda:write_json(Path(path),profiles),lambda _:self.message("Profiles saved."))

    def start(self):
        self.idle()
        if len(self.profiles)!=2:
            raise ValueError("Enroll or load both calibrated speakers")
        inp,out = self.device(self.input),self.device(self.output)
        options = (float(self.turn["male"].get()),float(self.turn["female"].get()),self.first.get(),
                   self.auto.get(),int(self.rounds.get()))
        live = LiveAudio(self.profiles,self.controller,float(self.fade.get()),self.record.get())
        Controller().command("start",options)
        live.volume = self.volume.get()
        self.cancel.clear()
        def started(_):
            self.live = live
            if self.cancel.is_set():
                self.stop()
                return
            self.controller.command("start",options)
            self.notice = f"Streaming at {live.rate} Hz."
        self.task("Opening full-duplex stream…",lambda:live.start(inp,out),started)

    def select(self,s):
        if self.live and self.controller.state().state in TURNS.values():
            self.auto.set(False)
            self.controller.command("select",s)

    def pause(self):
        if not self.live:
            return
        if self.controller.state().state=="PAUSED":
            tick = now_tick()
            self.controller.command("resume",tick=tick)
            self.live.play_after = tick
            self.live.public = True
        else:
            self.live.public = False
            self.controller.command("pause")

    def stop(self):
        if self.live and not self.job:
            live,self.live = self.live,None
            live.public = False
            self.controller.command("STOPPED")
            self.task("Closing stream…",live.close,
                      lambda _:self.message(live.error or "Stream stopped."))
        elif self.job:
            self.cancel.set()

    def quit(self):
        self.closing = True
        self.cancel.set()
        self.stop()

    def poll(self):
        if self.job and self.job[0].done():
            future,callback = self.job
            self.job = None
            try:
                callback(future.result())
            except Exception as exc:
                self.notice,self.failed = str(exc),True
                self.ready()
            self.progress = None
        if self.closing:
            if self.live:
                self.stop()
            if not self.job:
                self.pool.shutdown(wait=False)
                self.root.destroy()
                return
        live = self.live
        s = self.controller.state()
        note = self.notice
        if live:
            live.volume = self.volume.get()
            completed = live.end_cursor is not None and live.output.read >= live.end_cursor
            if live.error or not live.stream.active or completed:
                if not live.stream.active and not live.error:
                    live.error = "Device stream became inactive."
                self.stop()
                self.failed = bool(live.error)
            if live.latency_ms is None:
                budget = (sum(live.device_latency) + live.engine.delay/RATE +
                          live.down.delay_s + live.up.delay_s +
                          (len(live.input)+max(live.prefill,len(live.output)))/live.rate)
                latency = f"{1000*budget:.0f} ms budget"
            else:
                latency = f"{live.latency_ms:.0f} ms driver estimate"
            note += (f"  Latency {latency}. Underflows {live.underflows}, overflows {live.overflows}."
                     f"\nQueued input / output {len(live.input)} / {len(live.output)} samples.")
        if self.progress:
            note += f"  {self.progress[0]:.1f} s left"
        self.meters[0]["value"] = self.progress[1] if self.progress else live.input_level if live else 0
        self.meters[1]["value"] = live.output_level if live else 0
        route = s.route if s.state in TURNS.values() else "mute"
        suppressed = {"male":"female","female":"male","both":"neither","mute":"both"}[route]
        self.status.set(f"{s.state}   Authorized: {route}   Suppressed: {suppressed}   "
                        f"Round {s.turn//2+1}   {s.remaining/RATE:.1f} s\n{note}")
        self.profile_status.set("    ".join(f"{n.capitalize()}: {'enrolled' if n in self.profiles else 'needed'}" for n in SPEAKERS))
        self.label.config(fg="red" if self.failed else {"male":"#174d9b","female":"#68399b"}.get(route,"#414141"))
        self.root.after(75,self.poll)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
