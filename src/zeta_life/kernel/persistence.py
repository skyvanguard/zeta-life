"""PersistenceLayer -- identity persistence for the Conscious Kernel.

Provides save/load functionality for the entire kernel state, enabling
identity continuity across sessions.  Saves all nn.Module state_dicts,
non-weight state (latent vectors, step counters), and episodic memory
into a single checkpoint file with a human-readable JSON summary.

Usage::

    pl = PersistenceLayer(base_path='~/.zeta_life/')
    pl.save_state(components, identity_name='default')
    step = pl.load_state(components, identity_name='default')
    identities = pl.list_identities()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor


class PersistenceLayer:
    """Save and restore Conscious Kernel state for identity continuity.

    All checkpoints are stored under ``base_path`` as
    ``{identity_name}.ckpt`` (PyTorch binary) and
    ``{identity_name}.summary.json`` (human-readable metadata).

    Parameters
    ----------
    base_path : str
        Root directory for identity checkpoints.  Defaults to
        ``~/.zeta_life/`` and is created automatically if it does not exist.
    """

    def __init__(self, base_path: str = '~/.zeta_life/') -> None:
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_state(
        self,
        components: dict,
        identity_name: str = 'default',
    ) -> Path:
        """Persist all kernel components to disk.

        Parameters
        ----------
        components : dict
            Dictionary with the following keys:

            - ``world_model`` : :class:`WorldModel` (nn.Module)
            - ``self_model`` : :class:`SelfModel` (nn.Module)
            - ``error_engine`` : :class:`PredictionErrorEngine` (nn.Module)
            - ``fast_memory`` : :class:`FastMemory` (plain object with serialize)
            - ``slow_memory`` : :class:`SlowMemory` (nn.Module)
            - ``step`` : int
            - ``rssm_agent`` : optional, :class:`DreamerV3Agent` — present only
              when the kernel runs with ``world_model_type="rssm"``

        identity_name : str
            Name for the identity checkpoint (filename stem).

        Returns
        -------
        Path
            Path to the saved ``.ckpt`` file.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        step = components['step']

        # Collect state_dicts from all nn.Module components
        state_dicts: dict[str, dict] = {}
        nn_module_keys = [
            'world_model',
            'self_model',
            'error_engine',
            'slow_memory',
        ]
        for key in nn_module_keys:
            module = components[key]
            state_dicts[key] = module.state_dict()

        # Serialize fast memory (plain object, not nn.Module)
        fast_memory_data = components['fast_memory'].serialize()

        # Build checkpoint payload
        checkpoint = {
            'version': 1,
            'timestamp': timestamp,
            'step': step,
            'state_dicts': state_dicts,
            'fast_memory': fast_memory_data,
        }

        # RSSM agent (world_model_type="rssm"): networks + optimizers + EMA
        rssm_agent = components.get('rssm_agent')
        if rssm_agent is not None:
            checkpoint['rssm_agent'] = rssm_agent.state_dict()

        # Precision hyper-model (epistemic depth, when enabled)
        hypermodel = components.get('hypermodel')
        if hypermodel is not None:
            checkpoint['hypermodel'] = hypermodel.state_dict()

        # Write .ckpt (PyTorch binary)
        ckpt_path = self.base_path / f'{identity_name}.ckpt'
        torch.save(checkpoint, ckpt_path)

        # Write .summary.json (human-readable metadata)
        self_embedding = components['self_model'].self_embedding
        self_embedding_norm = float(torch.norm(self_embedding).item())

        summary = {
            'step': step,
            'timestamp': timestamp,
            'fast_memory_size': len(components['fast_memory']),
            'self_embedding_norm': self_embedding_norm,
        }

        summary_path = self.base_path / f'{identity_name}.summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return ckpt_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_state(
        self,
        components: dict,
        identity_name: str = 'default',
    ) -> int:
        """Restore kernel components from a saved checkpoint.

        Loads all nn.Module state_dicts back into their respective modules,
        restores fast memory from its serialized form, and returns the saved
        step counter.

        Parameters
        ----------
        components : dict
            Same structure as :meth:`save_state`.  The nn.Module instances
            **must** already be constructed with compatible architectures.
        identity_name : str
            Name of the identity to restore.

        Returns
        -------
        int
            The step counter from the checkpoint.

        Raises
        ------
        FileNotFoundError
            If no checkpoint exists for *identity_name*.
        """
        ckpt_path = self.base_path / f'{identity_name}.ckpt'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for identity '{identity_name}' at {ckpt_path}"
            )

        checkpoint = torch.load(ckpt_path, weights_only=False)

        # Restore nn.Module state_dicts
        nn_module_keys = [
            'world_model',
            'self_model',
            'error_engine',
            'slow_memory',
        ]
        for key in nn_module_keys:
            if key in checkpoint['state_dicts']:
                components[key].load_state_dict(checkpoint['state_dicts'][key])

        # Restore RSSM agent when both sides have one (older checkpoints
        # and GRU-mode kernels simply skip this block)
        rssm_agent = components.get('rssm_agent')
        if rssm_agent is not None and 'rssm_agent' in checkpoint:
            rssm_agent.load_state_dict(checkpoint['rssm_agent'])

        # Restore precision hyper-model when both sides have one
        hypermodel = components.get('hypermodel')
        if hypermodel is not None and 'hypermodel' in checkpoint:
            hypermodel.load_state_dict(checkpoint['hypermodel'])

        # Restore fast memory
        from zeta_life.kernel.complementary_memory import FastMemory
        restored_fm = FastMemory.restore(checkpoint['fast_memory'])
        # Copy the internal state into the existing FastMemory object
        fm = components['fast_memory']
        fm._episodes = restored_fm._episodes
        fm.capacity = restored_fm.capacity
        fm.surprise_threshold = restored_fm.surprise_threshold

        return checkpoint['step']

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_identities(self) -> list[str]:
        """Return names of all saved identities.

        Scans ``base_path`` for ``.ckpt`` files and returns their stems.

        Returns
        -------
        list[str]
            Sorted list of identity names.
        """
        ckpt_files = sorted(self.base_path.glob('*.ckpt'))
        return [p.stem for p in ckpt_files]

    def identity_exists(self, identity_name: str) -> bool:
        """Check whether a checkpoint exists for the given identity name.

        Parameters
        ----------
        identity_name : str
            Name to check.

        Returns
        -------
        bool
            ``True`` if a ``.ckpt`` file exists for *identity_name*.
        """
        return (self.base_path / f'{identity_name}.ckpt').exists()
