from dadra import sampling


def test_num_samples():
    assert sampling.num_samples(0.05, 1e-9, 3) == 941
