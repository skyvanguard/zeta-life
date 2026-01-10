:html_theme.sidebar_secondary.remove:

.. raw:: html

   <style>
   /* Hero Section */
   .hero-section {
       background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
       color: white;
       padding: 2.5rem 2rem;
       border-radius: 12px;
       margin-bottom: 2rem;
       text-align: center;
   }
   .hero-section h1 {
       color: white;
       margin-bottom: 0.5rem;
       font-size: 2.2rem;
   }
   .hero-section .tagline {
       color: #a0aec0;
       font-size: 1rem;
       margin-bottom: 1.5rem;
   }
   .hero-badges {
       display: flex;
       justify-content: center;
       gap: 0.5rem;
       flex-wrap: wrap;
       margin-bottom: 1.5rem;
   }
   .hero-badge {
       background: rgba(255,255,255,0.15);
       padding: 0.3rem 0.8rem;
       border-radius: 20px;
       font-size: 0.85rem;
       color: #63b3ed;
       border: 1px solid rgba(255,255,255,0.2);
   }
   .hero-buttons {
       display: flex;
       justify-content: center;
       gap: 1rem;
       flex-wrap: wrap;
   }
   .hero-btn {
       padding: 0.6rem 1.5rem;
       border-radius: 6px;
       text-decoration: none;
       font-weight: 500;
       font-size: 0.95rem;
       transition: all 0.2s;
   }
   .hero-btn-primary {
       background: #667eea;
       color: white !important;
   }
   .hero-btn-primary:hover {
       background: #5a6fd6;
   }
   .hero-btn-secondary {
       background: transparent;
       color: white !important;
       border: 1px solid rgba(255,255,255,0.3);
   }
   .hero-btn-secondary:hover {
       background: rgba(255,255,255,0.1);
   }

   /* Main Cards Grid */
   .main-cards {
       display: grid;
       grid-template-columns: repeat(2, 1fr);
       gap: 1.5rem;
       margin: 2rem 0;
   }
   @media (max-width: 768px) {
       .main-cards { grid-template-columns: 1fr; }
   }
   .main-card {
       border: 1px solid #e2e8f0;
       border-radius: 8px;
       padding: 1.5rem;
       transition: all 0.2s;
   }
   .main-card:hover {
       border-color: #667eea;
       box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
   }
   html[data-theme="dark"] .main-card {
       border-color: #4a5568;
       background: #1e293b;
   }
   .main-card h3 {
       margin: 0 0 0.5rem 0;
       font-size: 1.1rem;
   }
   .main-card h3 a {
       color: inherit;
       text-decoration: none;
   }
   .main-card h3 a:hover {
       color: #667eea;
   }
   .main-card p {
       color: #64748b;
       font-size: 0.9rem;
       margin: 0;
       line-height: 1.5;
   }
   html[data-theme="dark"] .main-card p {
       color: #94a3b8;
   }

   /* Results Section */
   .results-grid {
       display: grid;
       grid-template-columns: repeat(4, 1fr);
       gap: 1rem;
       margin: 1.5rem 0;
   }
   @media (max-width: 900px) {
       .results-grid { grid-template-columns: repeat(2, 1fr); }
   }
   @media (max-width: 500px) {
       .results-grid { grid-template-columns: 1fr; }
   }
   .result-card {
       text-align: center;
       padding: 1rem;
       background: #f8fafc;
       border-radius: 8px;
   }
   html[data-theme="dark"] .result-card {
       background: #1e293b;
   }
   .result-value {
       font-size: 1.8rem;
       font-weight: 700;
       color: #667eea;
   }
   .result-label {
       font-size: 0.8rem;
       color: #64748b;
   }

   /* Links Section */
   .useful-links {
       background: #f8fafc;
       padding: 1.5rem;
       border-radius: 8px;
       margin: 2rem 0;
   }
   html[data-theme="dark"] .useful-links {
       background: #1e293b;
   }
   .useful-links h3 {
       margin: 0 0 1rem 0;
       font-size: 1rem;
   }
   .links-grid {
       display: flex;
       flex-wrap: wrap;
       gap: 1rem 2rem;
   }
   .links-grid a {
       color: #667eea;
       text-decoration: none;
       font-size: 0.9rem;
   }
   .links-grid a:hover {
       text-decoration: underline;
   }
   </style>

   <div class="hero-section">
       <h1>Zeta-Life</h1>
       <p class="tagline">A research framework connecting Riemann zeta mathematics,<br>multi-agent emergence, and computational identity</p>

       <div class="hero-badges">
           <span class="hero-badge">93K+ Lines</span>
           <span class="hero-badge">72 Experiments</span>
           <span class="hero-badge">296 Tests</span>
           <span class="hero-badge">6/6 IPUESA</span>
       </div>

       <div class="hero-buttons">
           <a href="getting-started.html" class="hero-btn hero-btn-primary">Get Started</a>
           <a href="papers/zeta-life-framework-paper.html" class="hero-btn hero-btn-secondary">Read the Paper</a>
       </div>
   </div>


Zeta-Life Documentation
=======================

.. raw:: html

   <div class="main-cards">
       <div class="main-card">
           <h3><a href="getting-started.html">Getting Started</a></h3>
           <p>New to Zeta-Life? Start here for installation, basic concepts, and your first experiments with zeta-weighted kernels.</p>
       </div>
       <div class="main-card">
           <h3><a href="papers/zeta-life-framework-paper.html">Research Paper</a></h3>
           <p>The complete scientific paper with theoretical foundations, methodology, experimental results, and discussion.</p>
       </div>
       <div class="main-card">
           <h3><a href="EXPERIMENTS.html">Experiments</a></h3>
           <p>72 research scripts organized by domain: organism dynamics, consciousness, cellular automata, and validation.</p>
       </div>
       <div class="main-card">
           <h3><a href="api/modules.html">API Reference</a></h3>
           <p>Detailed documentation of all modules, classes, and functions in the zeta_life package.</p>
       </div>
   </div>


Key Results
-----------

.. raw:: html

   <div class="results-grid">
       <div class="result-card">
           <div class="result-value">+134%</div>
           <div class="result-label">Cell survival vs Moore</div>
       </div>
       <div class="result-card">
           <div class="result-value">11</div>
           <div class="result-label">Emergent properties</div>
       </div>
       <div class="result-card">
           <div class="result-value">±5%</div>
           <div class="result-label">Goldilocks zone</div>
       </div>
       <div class="result-card">
           <div class="result-value">6/6</div>
           <div class="result-label">IPUESA criteria</div>
       </div>
   </div>


The Zeta Kernel
---------------

At the core of Zeta-Life is the **zeta kernel**, derived from non-trivial zeros of the Riemann zeta function:

.. math::

   K_\sigma(t) = 2 \sum_{n=1}^{M} e^{-\sigma |\gamma_n|} \cos(\gamma_n t)

where :math:`\gamma_n` are the imaginary parts of zeta zeros (14.134, 21.022, 25.010, ...).

.. image:: _static/zeta_overview.png
   :alt: Zeta Kernel Overview
   :align: center
   :width: 100%


Quick Install
-------------

.. code-block:: bash

   pip install -e ".[full]"
   python -m pytest tests/ -v


.. raw:: html

   <div class="useful-links">
       <h3>Useful Links</h3>
       <div class="links-grid">
           <a href="getting-started.html">Installation</a>
           <a href="https://github.com/fruizvillar/zeta-life">Source Repository</a>
           <a href="https://github.com/fruizvillar/zeta-life/issues">Issue Tracker</a>
           <a href="REPRODUCE.html">Reproduce Results</a>
           <a href="TROUBLESHOOTING.html">Troubleshooting</a>
           <a href="GOLDILOCKS_DISCOVERY.html">Goldilocks Zone</a>
       </div>
   </div>


.. toctree::
   :maxdepth: 1
   :caption: Guide
   :hidden:

   Getting Started <getting-started>
   Reproduce <REPRODUCE>
   Troubleshooting <TROUBLESHOOTING>

.. toctree::
   :maxdepth: 1
   :caption: Research
   :hidden:

   Paper <papers/zeta-life-framework-paper>
   IPUESA <papers/ipuesa-identidad-funcional-paper>

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   Experiments <EXPERIMENTS>
   Theory <TEORIA_ZETA_EMERGENCIA>
   Goldilocks <GOLDILOCKS_DISCOVERY>
   API <api/modules>


Citation
--------

.. code-block:: bibtex

   @article{ruiz2026zetalife,
     title={Zeta-Life: A Unified Framework Connecting Riemann Zeta
            Mathematics, Multi-Agent Dynamics, and Functional Identity},
     author={Ruiz, Francisco},
     journal={arXiv preprint},
     year={2026}
   }


.. toctree::
   :hidden:

   GoL Docs <DOCUMENTACION>
