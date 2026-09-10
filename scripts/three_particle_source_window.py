"""Current-field PF evolution while a completed event's source remains alive."""
import numpy as np
from three_particle_forced_event import ContactEvent,MANIFEST
from three_particle_implicit_run import advance
from pf_sintering.three_particle_bounded_mobility import HarmonicSurfaceDiffusion
from pf_sintering.three_particle_event import ownership_pair_step


def advance_source_window(fields,seconds,g,pair,rule):
    local=ContactEvent(dict(g),pair);local.bind(fields);solver=HarmonicSurfaceDiffusion(local.op)
    current=tuple(x.copy() for x in fields);elapsed=0.;h=seconds
    while elapsed<seconds-1e-14:
        h=min(h,seconds-elapsed);local.bind(current)
        try:
            fn,error=advance(current[0],h/MANIFEST['seconds_per_model_time'],solver,rule)
            if error>1:raise RuntimeError('source window field error')
        except (FloatingPointError,RuntimeError):
            h*=.2
            if h<1e-10:raise RuntimeError('source window timestep floor')
            continue
        phi=ownership_pair_step(local.g['ownership'],fn,local.op,pair,h/MANIFEST['seconds_per_model_time'],1.0937500000000001e-25)
        current=(fn,*(phi*fn[None]));elapsed+=h
    if np.max(abs(sum(current[1:])-current[0]))>5e-15:raise RuntimeError('source window ownership closure')
    return current
