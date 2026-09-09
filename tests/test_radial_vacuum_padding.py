import numpy as np

from pf_sintering.radial_vacuum_padding import pad_case_radially_with_vacuum


def test_radial_padding_preserves_existing_state_and_coordinates_exactly():
    fields = np.arange(24.0).reshape(3, 8)
    geom = dict(f=fields, e1=0.25*fields, e2=0.75*fields,
                r_c=(np.arange(8)+0.5)*2.0, r_f=np.arange(9)*2.0,
                Nr=8, radial_margin_W=4.0)
    setup = dict(r_c=geom["r_c"].copy(), r_f=geom["r_f"].copy(),
                 dr=2.0, W=4.0)
    padded, new_setup, diagnostics = pad_case_radially_with_vacuum(
        geom, setup, target_outer_face_m=23.0)
    assert diagnostics["radial_padding_cells"] == 4
    assert diagnostics["new_radial_face_m"] == 24.0
    for key in ("f", "e1", "e2"):
        assert np.array_equal(padded[key][:, :8], geom[key])
        assert np.all(padded[key][:, 8:] == 0.0)
    assert np.array_equal(new_setup["r_c"][:8], setup["r_c"])
    assert np.array_equal(new_setup["r_f"][:9], setup["r_f"])
