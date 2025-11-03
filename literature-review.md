# CASI MATS — Task 2.1.5
### By: Parin Pengpun

## What I looked for

I focused on three areas:

1. **Decomposition / multi-turn jailbreaks** (how complex harmful goals are split into benign sub-requests).
2. **Compositional attacks / multi-model misuse** (how safe models can be combined to produce harm).
3. **Defenses** — especially **sequential / stateful monitors** and **weak-to-strong monitoring** proposals.

---

## Papers List

* **Monitoring Decomposition Attacks with Lightweight Sequential Monitors (June 2025)**
  Empirical paper showing that relatively lightweight sequential monitors can detect many decomposition attacks with good efficiency tradeoffs. *Why relevant:* provides a practical baseline for defenses we could attempt to reproduce/extend.
  Link: [https://arxiv.org/abs/2506.10949](https://arxiv.org/abs/2506.10949)

* **“Multi-Turn Jailbreaks Are Simpler Than They Seem” (Aug 2025)**
  Argues that multi-turn attacks sometimes gain only from more opportunities to retry rather than fundamentally new multi-turn mechanics. *Why relevant:* motivates careful experimental controls (e.g., compare multi-turn to repeated single-turn resampling) in our evaluations.
  Link: [https://arxiv.org/abs/2508.07646](https://arxiv.org/abs/2508.07646)

* **Foot-In-The-Door: A Multi-turn Jailbreak for LLMs (FITD, 2025)**
  Describes psychological-style sequential escalation attacks (useful for designing conversational baselines).
  Link: [https://arxiv.org/abs/2502.19820](https://arxiv.org/abs/2502.19820)

* **“Jailbreak in Pieces” (multi-modal compositional attacks, 2024)**
  Shows composition can cross modalities (image + text) and not just across text models. *Why relevant:* broadens scope of possible attack vectors we might need to consider.
  Link: [https://arxiv.org/abs/2402.09177](https://arxiv.org/abs/2402.09177)

* **Reliable Weak-to-Strong Monitoring of LLM Agents (Aug 2025)**
  Proposes methods to have weaker monitors reliably detect misbehavior of stronger agents. *Why relevant:* addresses a key practical limitation (cost/latency) of using frontier models as monitors.
  Link: [https://arxiv.org/abs/2508.19461](https://arxiv.org/abs/2508.19461)

* **Agentic red-teaming / automated attack generation (Genesis; Automatic LLM Red Teaming)**
  Recent work showing the automation of attack discovery and the risk of agentic, adaptive attackers. *Why relevant:* if our threat model includes automated attacker agents, we need to evaluate monitor robustness to adaptive strategies.
  Links: [https://arxiv.org/abs/2510.18314](https://arxiv.org/abs/2510.18314), [https://arxiv.org/abs/2508.04451](https://arxiv.org/abs/2508.04451)

---

## What the literature collectively suggests

* **Threat is real but nuanced.** Multiple independent papers show decomposition & composition attacks are *feasible* in realistic settings; BSD-style benchmarks provide empirical evidence. At the same time, there's evidence that increased attacker persistence (retries) explains part of multi-turn success, so experiments must control for that.
* **Defenses must be stateful or sequential to be meaningfully effective.** Stateless single-prompt filters are often blind to the trajectory of benign-appearing subtasks. Sequential monitors show promise, but they raise privacy and scalability tradeoffs.
* **The ecosystem perspective matters.** Safety-by-design for a single model is insufficient if other models (or tools) can be composed to finish the malicious task. Policy / governance complements technical defenses.
* **Adaptive, agentic attackers are the hardest case.** Monitors that only detect static signatures will be evaded by aware agents; monitoring strategies should be adaptive and consider game-theoretic dynamics.
