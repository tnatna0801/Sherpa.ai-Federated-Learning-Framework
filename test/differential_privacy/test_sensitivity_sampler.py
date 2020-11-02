import numpy as np

from shfl.private.query import Mean
from shfl.private.query import Query
from shfl.differential_privacy.probability_distribution import NormalDistribution
from shfl.differential_privacy import SensitivitySampler
from shfl.differential_privacy import L1SensitivityNorm
from shfl.differential_privacy import L2SensitivityNorm


def test_sample_sensitivity_gamma():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_m():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, m=285)

    assert np.abs(mean - 0) < 0.5


def test_sample_sensitivity_gamma_m():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L1SensitivityNorm(), distribution, n=100, m=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_l2_sensitivity_norm():
    distribution = NormalDistribution(0, 1)

    sampler = SensitivitySampler()
    _, mean = sampler.sample_sensitivity(Mean(), L2SensitivityNorm(), distribution, n=100, m=285, gamma=0.33)

    assert np.abs(mean - 0) < 0.5


def test_sensitivity_norm_list_of_arrays():
    class ReshapeToList(Query):
        def get(self, data):
            return list(np.reshape(data, (20, 30, 40)))

    distribution = NormalDistribution(0, 1.5)
    sampler = SensitivitySampler()

    # L1 norm:
    s_max, s_mean = sampler.sample_sensitivity(ReshapeToList(), L1SensitivityNorm(),
                                               distribution, n=20 * 30 * 40, m=285, gamma=0.33)

    assert isinstance(s_max, list)
    assert isinstance(s_mean, list)
    for i in range(len(s_max)):
        assert s_max[i].sum() < 2 * 1.5
        assert s_mean[i].sum() < 2 * 1.5

    # L2 norm:
    s_max, s_mean = sampler.sample_sensitivity(ReshapeToList(), L2SensitivityNorm(),
                                               distribution, n=20 * 30 * 40, m=285, gamma=0.33)

    assert isinstance(s_max, list)
    assert isinstance(s_mean, list)
    for i in range(len(s_max)):
        assert s_max[i].sum() < 2 * 1.5
        assert s_mean[i].sum() < 2 * 1.5
