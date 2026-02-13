# 🤖 Top 5 AI Papers This Week
## Week of February 13, 2026

Welcome to this week's roundup of the most impactful AI research papers! These papers have been generating buzz across Reddit, academic Twitter, and research communities.

**📊 This Week's Stats:**
- 📄 **5 featured papers** from **1 categories**  
- 👥 **56 contributing authors**
- 🔥 **Average engagement score:** 25.0
- 🏆 **Highest scorer:** 25 points

---

## 1. UniT: Unified Multimodal Chain-of-Thought Test-time Scaling

🧠 **Category:** CS.AI | 📅 **Published:** February 12, 2026 | 🔥 **Score:** 25 points

**Authors:** Leon Liangyu Chen, Haoyu Ma, Zhipeng Fan et al. (+11 more)

**Links:** [ArXiv Paper](https://arxiv.org/abs/2602.12279v1) | [PDF Download](https://arxiv.org/pdf/2602.12279v1.pdf)

Unified models can handle both multimodal understanding and generation within a single architecture, yet they typically operate in a single pass without iteratively refining their outputs.. Many multimodal tasks, especially those involving complex spatial compositions, multiple interacting objects, or evolving instructions, require decomposing instructions, verifying intermediate results, and making iterative corrections..

While test-time scaling (TTS) has demonstrated that allocating additional inference compute for iterative reasoning substantially improves language model performance, extending this paradigm to unified multimodal models remains an open challenge.. We introduce UniT, a framework for multimodal chain-of-thought test-time scaling that enables a single unified model to reason, verify, and refine across multiple rounds.. UniT combines agentic data synthesis, unified model training, and flexible test-time inference to elicit cognitive behaviors including verification, subgoal decomposition, and content memory..

Our key findings are: (1) unified models trained on short reasoning trajectories generalize to longer inference chains at test time; (2) sequential chain-of-thought reasoning provides a more scalable and compute-efficient TTS strategy than parallel sampling; (3) training on generation and editing trajectories improves out-of-distribution visual reasoning.. These results establish multimodal test-time scaling as an effective paradigm for advancing both generation and understanding in unified models..

---

## 2. Think like a Scientist: Physics-guided LLM Agent for Equation Discovery

🧠 **Category:** CS.AI | 📅 **Published:** February 12, 2026 | 🔥 **Score:** 25 points

**Authors:** Jianke Yang, Ohm Venkatachalam, Mohammad Kianezhad et al. (+2 more)

**Links:** [ArXiv Paper](https://arxiv.org/abs/2602.12259v1) | [PDF Download](https://arxiv.org/pdf/2602.12259v1.pdf)

Explaining observed phenomena through symbolic, interpretable formulas is a fundamental goal of science.. Recently, large language models (LLMs) have emerged as promising tools for symbolic equation discovery, owing to their broad domain knowledge and strong reasoning capabilities..

However, most existing LLM-based systems try to guess equations directly from data, without modeling the multi-step reasoning process that scientists often follow: first inferring physical properties such as symmetries, then using these as priors to restrict the space of candidate equations.. We introduce KeplerAgent, an agentic framework that explicitly follows this scientific reasoning process..

The agent coordinates physics-based tools to extract intermediate structure and uses these results to configure symbolic regression engines such as PySINDy and PySR, including their function libraries and structural constraints.. Across a suite of physical equation benchmarks, KeplerAgent achieves substantially higher symbolic accuracy and greater robustness to noisy data than both LLM and traditional baselines..

---

## 3. ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction

🧠 **Category:** CS.AI | 📅 **Published:** February 12, 2026 | 🔥 **Score:** 25 points

**Authors:** Nick Ferguson, Josh Pennington, Narek Beghian et al. (+4 more)

**Links:** [ArXiv Paper](https://arxiv.org/abs/2602.12247v1) | [PDF Download](https://arxiv.org/pdf/2602.12247v1.pdf)

Unstructured documents like PDFs contain valuable structured information, but downstream systems require this data in reliable, standardized formats.. LLMs are increasingly deployed to automate this extraction, making accuracy and reliability paramount..

However, progress is bottlenecked by two gaps.. First, no end-to-end benchmark evaluates PDF-to-JSON extraction under enterprise-scale schema breadth.. Second, no principled methodology captures the semantics of nested extraction, where fields demand different notions of correctness (exact match for identifiers, tolerance for quantities, semantic equivalence for names), arrays require alignment, and omission must be distinguished from hallucination.. We address both gaps with ExtractBench, an open-source benchmark and evaluation framework for PDF-to-JSON structured extraction.. The benchmark pairs 35 PDF documents with JSON Schemas and human-annotated gold labels across economically valuable domains, yielding 12,867 evaluatable fields spanning schema complexities from tens to hundreds of fields.. The evaluation framework treats the schema as an executable specification: each field declares its scoring metric.. Baseline evaluations reveal that frontier models (GPT-5/5.2, Gemini-3 Flash/Pro, Claude 4.5 Opus/Sonnet) remain unreliable on realistic schemas..

Performance degrades sharply with schema breadth, culminating in 0% valid output on a 369-field financial reporting schema across all tested models.. We release ExtractBench at https://github.com/ContextualAI/extract-bench..

---

## 4. DeepGen 1.0: A Lightweight Unified Multimodal Model for Advancing Image Generation and Editing

🧠 **Category:** CS.AI | 📅 **Published:** February 12, 2026 | 🔥 **Score:** 25 points

**Authors:** Dianyi Wang, Ruihang Li, Feng Han et al. (+17 more)

**Links:** [ArXiv Paper](https://arxiv.org/abs/2602.12205v1) | [PDF Download](https://arxiv.org/pdf/2602.12205v1.pdf)

Current unified multimodal models for image generation and editing typically rely on massive parameter scales (e.g., >10B), entailing prohibitive training costs and deployment footprints.. In this work, we present DeepGen 1.0, a lightweight 5B unified model that achieves comprehensive capabilities competitive with or surpassing much larger counterparts..

To overcome the limitations of compact models in semantic understanding and fine-grained control, we introduce Stacked Channel Bridging (SCB), a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable 'think tokens' to provide the generative backbone with structured, reasoning-rich guidance.. We further design a data-centric training strategy spanning three progressive stages: (1) Alignment Pre-training on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) Joint Supervised Fine-tuning on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) Reinforcement Learning with MR-GRPO, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts..

Despite being trained on only ~50M samples, DeepGen 1.0 achieves leading performance across diverse benchmarks, surpassing the 80B HunyuanImage by 28% on WISE and the 27B Qwen-Image-Edit by 37% on UniREditBench.. By open-sourcing our training code, weights, and datasets, we provide an efficient, high-performance alternative to democratize unified multimodal research..

---

## 5. Visual Reasoning Benchmark: Evaluating Multimodal LLMs on Classroom-Authentic Visual Problems from Primary Education

🧠 **Category:** CS.AI | 📅 **Published:** February 12, 2026 | 🔥 **Score:** 25 points

**Authors:** Mohamed Huti, Alasdair Mackintosh, Amy Waldock et al. (+7 more)

**Links:** [ArXiv Paper](https://arxiv.org/abs/2602.12196v1) | [PDF Download](https://arxiv.org/pdf/2602.12196v1.pdf)

AI models have achieved state-of-the-art results in textual reasoning; however, their ability to reason over spatial and relational structures remains a critical bottleneck -- particularly in early-grade maths, which relies heavily on visuals.. This paper introduces the visual reasoning benchmark (VRB), a novel dataset designed to evaluate Multimodal Large Language Models (MLLMs) on their ability to solve authentic visual problems from classrooms..

This benchmark is built on a set of 701 questions sourced from primary school examinations in Zambia and India, which cover a range of tasks such as reasoning by analogy, pattern completion, and spatial matching.. We outline the methodology and development of the benchmark which intentionally uses unedited, minimal-text images to test if models can meet realistic needs of primary education.. Our findings reveal a ``jagged frontier'' of capability where models demonstrate better proficiency in static skills such as counting and scaling, but reach a distinct ``spatial ceiling'' when faced with dynamic operations like folding, reflection, and rotation..

These weaknesses pose a risk for classroom use on visual reasoning problems, with the potential for incorrect marking, false scaffolding, and reinforcing student misconceptions.. Consequently, education-focused benchmarks like the VRB are essential for determining the functional boundaries of multimodal tools used in classrooms..

---


## 📈 About This Analysis

Each week, I analyze recent AI papers from ArXiv and rank them based on:

🗣️ **Social Media Engagement** - Mentions and discussions on Reddit  
🎯 **Research Impact Indicators** - Trending keywords and methodologies  
👥 **Collaboration Signals** - Author networks and institutional diversity  
⏰ **Recency Factor** - Boost for just-published papers  

**Methodology:** Papers are scored using a composite algorithm that weighs social media mentions (Reddit discussions, estimated Twitter activity) alongside content analysis for breakthrough keywords like "transformer," "multimodal," "reasoning," and others that typically indicate high-impact research.

**Coverage:** This analysis scans 7 major AI categories on ArXiv: Artificial Intelligence, Machine Learning, Natural Language Processing, Computer Vision, Neural Networks, Robotics, and Statistics ML.

---

*🤖 This analysis is automatically generated every Friday by monitoring ArXiv submissions and tracking social media engagement.*

**📬 Subscribe** for weekly AI research updates  
**💬 Share your thoughts** on this week's selections in the comments  
**🔗 Follow the project** on [GitHub](https://github.com/kjanik70/ai-papers-agent)

*Next edition: February 20, 2026*
