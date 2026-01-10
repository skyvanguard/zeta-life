:html_theme.sidebar_secondary.remove:

.. raw:: html

   <style>
   .hero-section {
       background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
       color: white;
       padding: 3rem 2rem;
       border-radius: 12px;
       margin-bottom: 2rem;
       text-align: center;
   }
   .hero-section h1 { color: white; margin-bottom: 1rem; }
   .hero-section p { color: #a0aec0; font-size: 1.1rem; }
   .stat-grid {
       display: grid;
       grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
       gap: 1rem;
       margin: 2rem 0;
   }
   .stat-box {
       background: rgba(255,255,255,0.1);
       padding: 1rem;
       border-radius: 8px;
       text-align: center;
   }
   .stat-number { font-size: 2rem; font-weight: bold; color: #63b3ed; }
   .stat-label { font-size: 0.85rem; color: #a0aec0; }
   .feature-grid {
       display: grid;
       grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
       gap: 1.5rem;
       margin: 2rem 0;
   }
   .feature-card {
       border: 1px solid #e2e8f0;
       border-radius: 8px;
       padding: 1.5rem;
       transition: all 0.2s;
   }
   .feature-card:hover {
       border-color: #667eea;
       box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
   }
   html[data-theme="dark"] .feature-card {
       border-color: #4a5568;
   }
   html[data-theme="dark"] .feature-card:hover {
       border-color: #667eea;
   }
   .feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
   .result-badge {
       display: inline-block;
       background: #48bb78;
       color: white;
       padding: 0.2rem 0.6rem;
       border-radius: 4px;
       font-size: 0.8rem;
       font-weight: 600;
   }
   </style>

   <div class="hero-section">
       <h1>Zeta-Life</h1>
       <p>A research framework unifying Riemann zeta mathematics,<br>multi-agent emergence, and computational functional identity</p>

       <div class="stat-grid">
           <div class="stat-box">
               <div class="stat-number">93K+</div>
               <div class="stat-label">Lines of Code</div>
           </div>
           <div class="stat-box">
               <div class="stat-number">72</div>
               <div class="stat-label">Experiments</div>
           </div>
           <div class="stat-box">
               <div class="stat-number">296</div>
               <div class="stat-label">Unit Tests</div>
           </div>
           <div class="stat-box">
               <div class="stat-number">6/6</div>
               <div class="stat-label">IPUESA Criteria</div>
           </div>
       </div>
   </div>

Key Results
===========

.. raw:: html

   <div class="feature-grid">
       <div class="feature-card">
           <div class="feature-icon">🎮</div>
           <h3>Zeta Game of Life</h3>
           <p>Cellular automata with zeta-weighted kernels show <strong>+134% cell survival</strong> compared to standard Moore neighborhood.</p>
           <span class="result-badge">p &lt; 0.001</span>
       </div>

       <div class="feature-card">
           <div class="feature-icon">🧬</div>
           <h3>11 Emergent Properties</h3>
           <p>ZetaOrganism exhibits homeostasis, regeneration, chemotaxis, collective panic, and more - <strong>without explicit programming</strong>.</p>
           <span class="result-badge">Verified</span>
       </div>

       <div class="feature-card">
           <div class="feature-icon">🎯</div>
           <h3>Goldilocks Zone</h3>
           <p>Functional identity exists only within a <strong>±5% parameter window</strong>. Identity is achievable but fragile.</p>
           <span class="result-badge">Key Finding</span>
       </div>

       <div class="feature-card">
           <div class="feature-icon">🔷</div>
           <h3>Abstract Vertices</h3>
           <p>Neutral V0-V3 tetrahedron replaces biased Jungian archetypes, enabling <strong>unbiased identity research</strong>.</p>
           <span class="result-badge">Novel Architecture</span>
       </div>
   </div>

The Zeta Kernel
---------------

At the heart of Zeta-Life is the **zeta kernel**, derived from the non-trivial zeros of the Riemann zeta function:

.. math::

   K_\sigma(t) = 2 \sum_{n=1}^{M} e^{-\sigma |\gamma_n|} \cos(\gamma_n t)

where :math:`\gamma_n` are the imaginary parts of zeta zeros (14.134725, 21.022040, 25.010858, ...).

This kernel provides a natural parameterization of the **"edge of chaos"** - the region where complex adaptive systems exhibit maximum computational capacity.

Quick Start
-----------

.. code-block:: bash

   # Install
   pip install -e ".[full]"

   # Run tests
   python -m pytest tests/ -v

   # Reproduce paper results
   python scripts/reproduce_paper.py

   # Run SYNTH-v2 experiment
   python experiments/consciousness/exp_ipuesa_synth_v2_consolidation.py


Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Main Paper

   papers/zeta-life-framework-paper

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   REPRODUCE
   EXPERIMENTS
   TROUBLESHOOTING
   GOLDILOCKS_DISCOVERY

.. toctree::
   :maxdepth: 2
   :caption: Theory

   TEORIA_ZETA_EMERGENCIA
   DOCUMENTACION

.. toctree::
   :maxdepth: 2
   :caption: Research Papers

   papers/ipuesa-identidad-funcional-paper

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Design Documents

   plans/2026-01-09-abstract-vertices-design


IPUESA Methodology
==================

The **IPUESA** (Investigation of Universal Persistence of Entities with Anticipatory Self) methodology operationalizes "functional identity" through 6 measurable metrics:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Metric
     - Description
     - Threshold
     - SYNTH-v2 Result
   * - **HS**
     - Holographic Survival
     - 0.30-0.70
     - 0.401 ✓
   * - **TAE**
     - Temporal Anticipation Effectiveness
     - ≥ 0.15
     - 0.217 ✓
   * - **MSR**
     - Module Spreading Rate
     - ≥ 0.15
     - 0.498 ✓
   * - **EI**
     - Embedding Integrity
     - ≥ 0.30
     - 0.996 ✓
   * - **ED**
     - Emergent Differentiation
     - ≥ 0.10
     - 0.361 ✓
   * - **deg_var**
     - Degradation Variance
     - ≥ 0.02
     - 0.028 ✓


Citation
--------

If you use Zeta-Life in your research, please cite:

.. code-block:: bibtex

   @article{ruiz2026zetalife,
     title={Zeta-Life: A Unified Framework Connecting Riemann Zeta Mathematics,
            Multi-Agent Dynamics, and Computational Functional Identity},
     author={Ruiz, Francisco},
     journal={arXiv preprint},
     year={2026}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
