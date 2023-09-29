from satio_pc.layers import _basenames, _fn

def test_layers_paths():
    for k in _basenames.keys():
        assert _fn(k).is_file()