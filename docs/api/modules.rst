API Reference
=============

This section provides an overview of the Zeta-Life Python API.

Core Module (``zeta_life.core``)
--------------------------------

The core module contains the mathematical foundations:

* **ZetaKernel**: Kernel function using Riemann zeta zeros
* **ZetaPsyche**: Jungian archetype-based consciousness model
* **ZetaMemorySystem**: Episodic, semantic, and procedural memory
* **ZetaLSTM**: LSTM with zeta temporal memory

Key Classes
^^^^^^^^^^^

``ZetaKernel``
    Implements the kernel :math:`K_\sigma(t) = 2 \sum_n e^{-\sigma|\gamma_n|} \cos(\gamma_n t)`
    where :math:`\gamma_n` are the imaginary parts of zeta zeros.

``ZetaPsyche``
    Four-archetype system in tetrahedral space:

    - PERSONA: Social mask
    - SOMBRA: Shadow/unconscious
    - ANIMA: Receptive/emotional
    - ANIMUS: Active/rational

Organism Module (``zeta_life.organism``)
----------------------------------------

Multi-agent emergent intelligence system:

* **ZetaOrganism**: Main simulation orchestrator
* **CellState**: Cell states (MASS, FORCE, CORRUPT)
* **ForceField**: Zeta-kernel force field
* **BehaviorEngine**: Neural network for A-B influence

Demonstrated emergent properties:

- Homeostasis, Regeneration, Antifragility
- Quimiotaxis, Spatial memory, Auto-segregation
- Competitive exclusion, Niche partition
- Collective panic, Coordinated escape

Psyche Module (``zeta_life.psyche``)
------------------------------------

Jungian consciousness implementation:

* **ZetaPsyche**: Archetype navigation
* **ZetaIndividuation**: Self integration (8 stages)
* **AttractorMemory**: Converged state storage
* **OrganicVoice**: Internal perspective descriptions

Consciousness Module (``zeta_life.consciousness``)
--------------------------------------------------

Hierarchical consciousness architecture:

* **MicroPsyche**: Cell-level psyche
* **Cluster**: Cluster aggregation
* **OrganismConsciousness**: Organism-level integration
* **BottomUpIntegrator**: Cell->Cluster->Organism flow
* **TopDownModulator**: Organism->Cluster->Cell influence

Cellular Module (``zeta_life.cellular``)
----------------------------------------

Zeta Game of Life cellular automata:

* **ZetaGameOfLife**: Basic cellular automaton
* **ZetaNeuralCA**: Differentiable Neural CA
* **ZetaRNN**: LSTM with zeta temporal memory

Utils Module (``zeta_life.utils``)
----------------------------------

Shared utilities:

* **statistics**: Confidence intervals, p-values, effect sizes
* **visualization**: Plotting helpers
* **config**: Configuration management

Usage Example
-------------

.. code-block:: python

    from zeta_life.core import ZetaKernel, ZetaPsyche

    # Create kernel with 15 zeta zeros
    kernel = ZetaKernel(M=15, sigma=0.1)

    # Evaluate kernel at time t=1.0
    value = kernel(1.0)

    # Create psyche with 4 archetypes
    psyche = ZetaPsyche(embedding_dim=8)

    # Process a stimulus
    response = psyche.process_stimulus("exploration")

For more detailed examples, see the :doc:`../EXPERIMENTS` page.
