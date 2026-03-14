"""Tests for the datasets module: sources, projector, and adapter."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from zeta_life.datasets import (
    ColoredNoiseSource,
    CSVSource,
    DatasetAdapter,
    Projector,
    SyntheticSignalSource,
)


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------


class TestProjector:
    def test_pca_output_shape(self):
        data = np.random.default_rng(0).standard_normal((100, 8))
        proj = Projector(target_dim=4, method="pca").fit(data)
        out = proj.transform(data[0])
        assert out.shape == (4,)

    def test_pca_batch_shape(self):
        data = np.random.default_rng(0).standard_normal((100, 8))
        proj = Projector(target_dim=4, method="pca").fit(data)
        out = proj.transform(data[:10])
        assert out.shape == (10, 4)

    def test_identity_passthrough(self):
        data = np.random.default_rng(0).standard_normal((50, 4))
        proj = Projector(target_dim=4, method="identity").fit(data)
        sample = data[0]
        out = proj.transform(sample)
        np.testing.assert_allclose(out, sample, atol=1e-10)

    def test_identity_dimension_mismatch_raises(self):
        data = np.random.default_rng(0).standard_normal((50, 8))
        with pytest.raises(ValueError, match="n_features == target_dim"):
            Projector(target_dim=4, method="identity").fit(data)

    def test_explained_variance_sums_to_one_or_less(self):
        data = np.random.default_rng(0).standard_normal((200, 10))
        proj = Projector(target_dim=4, method="pca").fit(data)
        ratio = proj.explained_variance_ratio
        assert ratio is not None
        assert ratio.sum() <= 1.0 + 1e-10
        assert all(r >= 0 for r in ratio)

    def test_fit_returns_self(self):
        data = np.random.default_rng(0).standard_normal((50, 4))
        proj = Projector(target_dim=4)
        result = proj.fit(data)
        assert result is proj

    def test_transform_before_fit_raises(self):
        proj = Projector(target_dim=4)
        with pytest.raises(RuntimeError, match="fit"):
            proj.transform(np.zeros(4))


# ---------------------------------------------------------------------------
# ColoredNoiseSource
# ---------------------------------------------------------------------------


class TestColoredNoise:
    def test_load_shape(self):
        src = ColoredNoiseSource(n_samples=1000, n_channels=4)
        data = src.load()
        assert data.shape == (1000, 4)

    def test_reproducibility(self):
        a = ColoredNoiseSource(n_samples=500, seed=42).load()
        b = ColoredNoiseSource(n_samples=500, seed=42).load()
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = ColoredNoiseSource(n_samples=500, seed=1).load()
        b = ColoredNoiseSource(n_samples=500, seed=2).load()
        assert not np.allclose(a, b)

    def test_spectral_slope_pink(self):
        """Pink noise should have ~1/f spectrum (beta=1)."""
        src = ColoredNoiseSource(n_samples=8192, n_channels=1, noise_type="pink", seed=0)
        data = src.load()[:, 0]
        freqs = np.fft.rfftfreq(len(data))[1:]  # skip DC
        spectrum = np.abs(np.fft.rfft(data))[1:]
        # Log-log regression: slope should be approximately -0.5 (amplitude ~ f^{-beta/2})
        log_f = np.log10(freqs)
        log_s = np.log10(spectrum + 1e-20)
        slope = np.polyfit(log_f, log_s, 1)[0]
        assert slope < 0, "Pink noise should have negative spectral slope"

    def test_mixed_type(self):
        src = ColoredNoiseSource(n_samples=500, n_channels=4, noise_type="mixed")
        data = src.load()
        assert data.shape == (500, 4)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="noise_type"):
            ColoredNoiseSource(noise_type="blue")

    def test_protocol_properties(self):
        src = ColoredNoiseSource(n_samples=100, n_channels=3)
        assert src.name == "colored_noise_pink"
        assert src.n_features == 3
        assert src.n_samples == 100
        assert len(src.feature_names) == 3


# ---------------------------------------------------------------------------
# SyntheticSignalSource
# ---------------------------------------------------------------------------


class TestSyntheticSignal:
    def test_load_shape(self):
        src = SyntheticSignalSource(n_samples=1000, n_channels=4)
        data = src.load()
        assert data.shape == (1000, 4)

    def test_reproducibility(self):
        a = SyntheticSignalSource(seed=42).load()
        b = SyntheticSignalSource(seed=42).load()
        np.testing.assert_array_equal(a, b)

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="pattern"):
            SyntheticSignalSource(pattern="invalid")


# ---------------------------------------------------------------------------
# CSVSource
# ---------------------------------------------------------------------------


class TestCSVSource:
    def _write_csv(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "test.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_load_basic(self, tmp_path):
        p = self._write_csv(tmp_path, "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        src = CSVSource(p, normalize=False)
        data = src.load()
        assert data.shape == (3, 3)
        np.testing.assert_allclose(data[0], [1, 2, 3])

    def test_column_selection_by_name(self, tmp_path):
        p = self._write_csv(tmp_path, "x,y,z\n1,2,3\n4,5,6\n")
        src = CSVSource(p, columns=["x", "z"], normalize=False)
        data = src.load()
        assert data.shape == (2, 2)
        np.testing.assert_allclose(data[0], [1, 3])

    def test_column_selection_by_index(self, tmp_path):
        p = self._write_csv(tmp_path, "a,b,c\n1,2,3\n4,5,6\n")
        src = CSVSource(p, columns=[0, 2], normalize=False)
        data = src.load()
        np.testing.assert_allclose(data[0], [1, 3])

    def test_normalize(self, tmp_path):
        p = self._write_csv(tmp_path, "a,b\n0,0\n10,10\n")
        src = CSVSource(p, normalize=True)
        data = src.load()
        # z-scored: mean~0, std~1
        np.testing.assert_allclose(data.mean(axis=0), 0, atol=1e-10)

    def test_feature_names_from_header(self, tmp_path):
        p = self._write_csv(tmp_path, "open,high,low,close\n1,2,3,4\n")
        src = CSVSource(p, normalize=False)
        src.load()
        assert src.feature_names == ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------


class TestDatasetAdapter:
    def test_stimulus_shape(self):
        src = ColoredNoiseSource(n_samples=100, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        stim, phase = adapter.get_stimulus()
        assert stim.shape == (4,)
        assert isinstance(stim, torch.Tensor)

    def test_stimulus_positive_values(self):
        src = ColoredNoiseSource(n_samples=200, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        for _ in range(50):
            stim, _ = adapter.get_stimulus()
            assert (stim > 0).all(), f"Expected all positive, got {stim}"

    def test_hierarchical_stimulus_shape(self):
        src = ColoredNoiseSource(n_samples=100, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        stim = adapter.get_hierarchical_stimulus()
        assert isinstance(stim, np.ndarray)
        assert stim.shape == (4,)

    def test_loop_wraps_around(self):
        src = ColoredNoiseSource(n_samples=10, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity", loop=True)
        # Consume all 10 + 5 more (wraps)
        for _ in range(15):
            adapter.get_stimulus()

    def test_no_loop_raises(self):
        src = ColoredNoiseSource(n_samples=5, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity", loop=False)
        for _ in range(5):
            adapter.get_stimulus()
        with pytest.raises(StopIteration):
            adapter.get_stimulus()

    def test_reset(self):
        src = ColoredNoiseSource(n_samples=50, n_channels=4, seed=0)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        first, _ = adapter.get_stimulus()
        adapter.reset()
        again, _ = adapter.get_stimulus()
        torch.testing.assert_close(first, again)

    def test_phase_labels(self):
        src = ColoredNoiseSource(n_samples=100, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        phases_seen = set()
        for _ in range(100):
            _, phase = adapter.get_stimulus()
            phases_seen.add(phase)
        assert phases_seen == {"early", "mid", "late", "final"}

    def test_len(self):
        src = ColoredNoiseSource(n_samples=200, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        assert len(adapter) == 200

    def test_pca_projection_higher_dim(self):
        """Source with more features than obs_dim uses PCA."""
        src = ColoredNoiseSource(n_samples=500, n_channels=8)
        adapter = DatasetAdapter(src, obs_dim=4, projection="pca")
        stim, _ = adapter.get_stimulus()
        assert stim.shape == (4,)

    def test_compatible_with_kernel(self):
        """Stimulus works with ConsciousKernel.step()."""
        from zeta_life.kernel.conscious_kernel import ConsciousKernel

        src = ColoredNoiseSource(n_samples=100, n_channels=4)
        adapter = DatasetAdapter(src, obs_dim=4, projection="identity")
        kernel = ConsciousKernel(obs_dim=4)

        stim, _ = adapter.get_stimulus()
        result = kernel.step(stim)
        assert result.free_energy >= 0.0
