# 2.1.6 — Future work (CASI MATS)
### By: Parin Pengpun

This fields feels like being at the beginning of something huge and weird, the kind of work that makes you wake up at 3 a.m. with half-formed experiments in your head. I’m excited and anxious in equal measure. 

## Excitements

* The *research space is enormous*. We’re only starting to map how small, harmless-seeming subrequests can be stitched into harmful flows. That means low-hanging fruit: reproducible experiments, better benchmarks, clearer threat taxonomies.

* Interpretability feels like a gold mine. Techniques from interpretability (sparse activations, mechanistic probes, etc.) give us glimpses into how models break down tasks internally.

* The interdisciplinary angle. This also involves law, economics, sociology, human factors. I love that a good solution to this problem generally probably won’t be purely technical, it will be socio-technical, which means collaborating outside CS and learning a lot fast.

## Worries

* **If a system gets superintelligent, will we even be able to edit its “brain”? Or put it on safeguards?** If models scale in ways we can’t introspect, the idea of surgical fixes, “change this neuron, remove that behavior”, might be fantasy. What does “editability” mean at scale? If editability drops as capability rises, our defenses might be transient.
* **Governance vs. geopolitics.** Governance can be thoughtful, principled, well-designed and still fail if a single powerful actor sees an advantage in ignoring it. How do we create incentives that are robust to bad actors and cross-border competition? I worry less about producing rules on paper and more about alignment between incentives, economics, and institutions.
* **Guarantees are illusory.** We can harden systems, build monitors, and design audits — but “guaranteed alignment” reads to me like a political promise more than a technical proof. That doesn’t mean we shouldn’t try; it means we should be honest about residual risk and design for graceful failure modes.
* **Concentration of power.** If capability accumulates with a few providers or states, then control over alignment becomes a political concentration problem. Tech fixes won’t solve concentration; that’s a governance, market, and policy problem too.

## Questions I have

* What *does* count as an acceptable mitigation? Is it enough to reduce success rates of attacks from 90% to 5%? Or do we need near-zero for specific risk classes?
* How much does the attacker model matter? If a human attacker can coordinate models across providers, how does that change our defensive posture versus a single-agent attacker?
* Can we design monitoring that is both effective and privacy-preserving? Is there a sweet spot where we log enough to detect long-term composition attacks without creating a surveillance system?
* Are we chasing the wrong level of abstraction? Maybe instead of trying to stop every possible misuse we should identify the *few* failure modes that cause truly catastrophic outcomes and harden around those. How do we find and prove those?
* Finally: who should decide what “harmful” means in cross-cultural and cross-national contexts? The technical work will be useless if stakeholders don’t trust the definitions and process.

## Promising directions I want to work on

* **Adaptive red-teaming.** Evaluate defenses against attackers that learn about the monitor and adapt. Simple static tests are necessary but insufficient; robustness to an adaptive adversary feels critical.
* **Cross-provider compositional tests.** Build small, reproducible experiments that stitch together different publicly-available models in short pipelines so we can measure compositional risk in practice. This is pragmatic: real attackers will mix and match.
* **Mechanistic-guided monitors.** Use interpretability signals (e.g., activation patterns, concept detectors) to build lighter-weight, explainable monitors that flag suspicious trajectories. If a monitor can point to why it flagged a conversation, that helps triage and builds trust.