import unittest

import numpy as np

from src.debate_controller import Controller, RATE, Router


class DebateTests(unittest.TestCase):
    def setUp(self):
        self.c = Controller(lambda:0)
        self.c.command("start",(3,4,"male",True,0))

    def test_schedule_sample_edges_no_drift_and_rounds(self):
        for cycle in (0,1,117,20000):
            start = 7*cycle*RATE
            self.assertEqual(self.c.state(start).turn,2*cycle)
            self.assertEqual(self.c.state(start+3*RATE-1).speaker,"male")
            self.assertEqual(self.c.state(start+3*RATE).speaker,"female")
        self.c.command("start",(3,4,"female",True,2))
        self.assertEqual(self.c.state(7*RATE-1).speaker,"male")
        self.assertEqual(self.c.state(14*RATE).state,"STOPPED")

    def test_capture_time_routing_pause_resume_and_restart(self):
        self.c.command("pause",tick=RATE)
        self.assertEqual(self.c.state(10*RATE).remaining,2*RATE)
        self.c.command("resume",tick=11*RATE)
        self.assertEqual(self.c.state(13*RATE).speaker,"female")
        spans = self.c.spans(RATE-1,10*RATE+2)
        self.assertEqual([(s.state,n) for s,n in spans],[("MALE_TURN",1),("PAUSED",10*RATE),("MALE_TURN",1)])
        self.c.command("STOPPED",tick=15*RATE)
        router = Router(self.c,0)
        out,_ = router.apply(15*RATE,np.ones((3,2)),np.ones(3))
        self.assertFalse(out.any())
        self.c.command("start",(3,4,"female",True,0),tick=16*RATE)
        self.assertEqual(self.c.state(16*RATE).turn,0)

    def test_manual_both_mute_and_next(self):
        router = Router(self.c,0)
        for i,(route,want) in enumerate((("male",2),("both",4),("mute",0),("female",1))):
            self.c.command("select",route,tick=i*RATE)
            out,_ = router.apply(i*RATE,np.tile([1,2],(4,1)),np.full(4,4))
            np.testing.assert_equal(out,want)
        self.assertFalse(self.c.state(100*RATE).auto)
        self.c.command("next",tick=101*RATE)
        self.assertEqual(self.c.state(101*RATE).speaker,"male")

    def test_crossfade_spans_blocks(self):
        router = Router(self.c,20)
        x = np.zeros(2*RATE)
        y = np.column_stack((x,x+1))
        a,_ = router.apply(2*RATE,y[:RATE+7],x[:RATE+7])
        b,_ = router.apply(3*RATE+7,y[:400],x[:400])
        transition = np.r_[a[RATE:],b]
        np.testing.assert_allclose(transition[:320],1-np.arange(1,321)/320,atol=1e-7)
        self.assertFalse(transition[320:].any())
        for duration in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                self.c.command("start",(duration,4,"male",True,0))
