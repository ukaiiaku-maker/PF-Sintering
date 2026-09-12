"""One-time symmetric handoff from paired half-domain data.

The returned arrays occupy the ordinary full z domain.  This module performs
no evolution-time projection or constraint.
"""
import numpy as np


def reconstruct_full_state_from_paired_halves(f, ownership):
    """Return the least-change symmetric full field and closed grain fields.

    Opposite half-domain samples are averaged once.  Outer-grain labels swap
    under reflection; the center label does not.  The three eta fields close
    algebraically to f on the retained half and are then reflected explicitly.
    """
    f=np.asarray(f);ownership=np.asarray(ownership)
    if f.ndim!=2 or f.shape[0]%2:raise ValueError('requires an even full z grid')
    if ownership.shape!=(3,*f.shape):raise ValueError('ownership shape mismatch')
    mid=f.shape[0]//2
    left=.5*(f[:mid]+f[mid:][::-1])
    phi_left=np.empty((3,*left.shape),dtype=np.result_type(f,ownership))
    phi_left[0]=.5*(ownership[0,:mid]+ownership[2,mid:][::-1])
    phi_left[1]=.5*(ownership[1,:mid]+ownership[1,mid:][::-1])
    phi_left[2]=.5*(ownership[2,:mid]+ownership[0,mid:][::-1])
    total=phi_left.sum(axis=0)
    if np.any(total<=0) or not np.isfinite(total).all():raise ValueError('invalid ownership simplex')
    phi_left/=total
    eta_left=np.empty_like(phi_left)
    eta_left[0]=left*phi_left[0]
    eta_left[1]=left*phi_left[1]
    eta_left[2]=left-eta_left[0]-eta_left[1]
    full_f=np.concatenate((left,left[::-1]),axis=0)
    eta=np.empty((3,*full_f.shape),dtype=eta_left.dtype)
    eta[:,:mid]=eta_left
    eta[0,mid:]=eta_left[2,::-1]
    eta[1,mid:]=eta_left[1,::-1]
    eta[2,mid:]=eta_left[0,::-1]
    phi=np.divide(eta,full_f[None],out=np.empty_like(eta),where=abs(full_f[None])>1e-30)
    tiny=abs(full_f)<=1e-30
    phi[:,tiny]=np.concatenate((phi_left,phi_left[[2,1,0],::-1]),axis=1)[:,tiny]
    return full_f,phi,(full_f,*eta)
