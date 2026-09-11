"""Two independent production root clocks and field-resolved first crossing.

Both clocks pause while a serialized avalanche/source is active. At extinction
only the activated contact resets and draws a new ordinary production threshold;
the other retains its hazard and threshold. No alternation or simultaneous roots.
"""
import copy
import numpy as np

CONTACTS=('LEFT','RIGHT')


def cumulative_event_quota(event_number,phase,q_over_b):
    """Bicrystal qcum accounting across serialized starts, failures and windows."""
    if event_number < 0 or int(event_number) != event_number:
        raise ValueError('invalid event number')
    q=float(q_over_b)
    if not np.isfinite(q) or not 0 <= q <= 1+1e-12:
        raise ValueError('invalid active event quota')
    active={'ACTIVE_ONE_B','ONE_B_COMPLETE','ONE_B_FAILED','ACTIVE_EVENT_RESUMED'}
    inactive={'POST_TRANSIENT_NEW_TRAJECTORY','RELOAD','ROOT_CROSSING',
              'SOURCE_WINDOW_OPEN','SOURCE_WINDOW','FACILITATED_WINDOW',
              'CHILD_CROSSING','AVALANCHE_EXTINCT_REPINNED','FORCED_ROOT_COMPLETE'}
    if phase in active:
        if event_number < 1:raise ValueError('active quota requires an event')
        return float(event_number-1)+q
    if phase not in inactive:raise ValueError('unknown quota accounting phase: '+phase)
    return float(event_number)

class RootClocks:
    def __init__(self,rng,multiplier=1.25):
        self.rng=rng;self.multiplier=float(multiplier)
        self.hazard={c:0. for c in CONTACTS}
        self.threshold={c:self.multiplier*float(rng.exponential()) for c in CONTACTS}
        self.active=None;self.avalanche_id=0
    def snapshot(self):
        return dict(hazard=self.hazard.copy(),threshold=self.threshold.copy(),active=self.active,
                    avalanche_id=self.avalanche_id,multiplier=self.multiplier,rng=copy.deepcopy(self.rng.bit_generator.state))
    @classmethod
    def restore(cls,saved):
        obj=cls.__new__(cls);obj.rng=np.random.default_rng();obj.rng.bit_generator.state=copy.deepcopy(saved['rng'])
        obj.multiplier=saved['multiplier'];obj.hazard=saved['hazard'].copy();obj.threshold=saved['threshold'].copy()
        obj.active=saved['active'];obj.avalanche_id=saved['avalanche_id'];return obj
    def increments(self,rates0,rates1,dt_s):
        if self.active is not None:raise RuntimeError('root clocks frozen while source alive')
        if dt_s<=0:raise ValueError('positive clock interval required')
        if any(not np.isfinite(r[c]) or r[c]<0 for r in (rates0,rates1) for c in CONTACTS):raise ValueError('invalid root rate')
        return {c:.5*(rates0[c]+rates1[c])*dt_s for c in CONTACTS}
    def crossed(self,increments):
        return [c for c in CONTACTS if self.hazard[c]+increments[c]>=self.threshold[c]]
    def commit(self,increments,contact=None):
        if self.active is not None:raise RuntimeError('root clock already committed')
        crossing=self.crossed(increments)
        if contact is None and crossing:raise RuntimeError('crossing must be localized before commit')
        if contact is not None and contact not in crossing:raise RuntimeError('selected root has not crossed')
        for c in CONTACTS:self.hazard[c]+=increments[c]
        if contact is not None:
            self.hazard[contact]=self.threshold[contact];self.active=contact;self.avalanche_id+=1
    def extinct(self):
        if self.active is None:raise RuntimeError('no active source')
        c=self.active;self.hazard[c]=0.;self.threshold[c]=self.multiplier*float(self.rng.exponential());self.active=None


def locate_first_root(field,dt_s,advance_s,rates,clocks,tolerance_s):
    """Reintegrate the complete field from the accepted left endpoint.

    Caller supplies an error-controlled field integrator in physical seconds.
    No trial mutates clocks or RNG. The returned upper-bracket field is within
    tolerance_s of the first trapezoidal hazard crossing. An exact numeric tie
    uses LEFT as a deterministic serialization rule, never a forced alternation.
    """
    if tolerance_s<=0:raise ValueError('positive crossing tolerance required')
    r0=rates(field);end=advance_s(field,dt_s);inc=clocks.increments(r0,rates(end),dt_s)
    if not clocks.crossed(inc):return end,dt_s,inc,None
    lo=0.;hi=dt_s
    while hi-lo>tolerance_s:
        mid=.5*(lo+hi);trial=advance_s(field,mid)
        candidate=clocks.increments(r0,rates(trial),mid)
        if clocks.crossed(candidate):hi=mid;end=trial;inc=candidate
        else:lo=mid
    # The upper endpoint retained above is always an actually evolved field.
    candidates=clocks.crossed(inc)
    selected=min(candidates,key=lambda c:((clocks.threshold[c]-clocks.hazard[c])/inc[c],CONTACTS.index(c)))
    return end,hi,inc,selected
