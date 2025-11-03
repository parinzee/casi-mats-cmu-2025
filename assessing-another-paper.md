# 2.1.3: Assessing Glukhov et al. [Glu+24]

## What unique value does it add beyond previous work?

After reading the prior work, I think Glukhov et al. makes three core contributions that the field was missing:

**1. A formal framework for measuring information leakage**

Jones et al. [JDS24] showed decomposition attacks work empirically, they got 43% success rates exploiting model combinations. Brown et al. [Bro+25] built better benchmarks and measured detectability. However, both used binary metrics: did the attack succeed? Was it detected?

Glukhov introduces Impermissible Information Leakage (IIL), which measures *how much* an adversary learns in bits. Instead of "attack succeeded or failed," they quantify "adversary gained X bits of information about dangerous knowledge." By making a binary metric to continuous measurement lets us compare defenses more rigorously and reason about cumulative leakage across multiple queries.

**2. Reframing the adversary model**

Traditionally, jailbreaking assumes adversaries want to *force harmful outputs* (what they call "security adversaries"). But decomposition attacks work because adversaries can *infer impermissible information from benign outputs* ("inferential adversaries").

This explains something important that prior work showed but didn't formalize: why aligned models that refuse harmful queries are still vulnerable. The model never outputs "here's how to make a bioweapon," but by answering questions about culture media, spore formation, and aerosolization separately, an adversary gains the knowledge anyway. 

Brown et al. measured this empirically (decomposition gets 46-68% accuracy while staying undetected), but Glukhov explains *why* through information theory: benign individual responses compose into impermissible knowledge.

**3. Proving fundamental impossibility results**

This is the most important theoretical contribution. Through their safety-utility tradeoff theorems (5.6, 5.7), they prove that you *cannot* simultaneously:
- Bound information leakage below some threshold
- Preserve full utility for legitimate users
- Defend against adaptive adversaries

This isn't an engineering problem we might solve with better techniques, but it's mathematic. Any defense that bounds information leakage must either refuse legitimate queries or degrade response quality.

I find this result particularly valuable because it tells us to stop looking for perfect solutions and start optimizing the tradeoff. 

## What does their mathematical framework accomplish?

The framework provides three main tools:

**IIL (Impermissible Information Leakage):** Measures how much an adversary's belief about dangerous knowledge shifts after seeing model responses. Formally: IIL = p(correct|responses) × log(p(correct|responses)/p(correct|prior)). This calibrates by the adversary's final confidence, so gaining certainty matters more than just shifting probabilities slightly.

**Exp-IIL and composition bounds:** They show how information leakage accumulates across multiple queries. This is critical because decomposition attacks inherently involve many queries. The composability theorem (5.3) provides worst-case bounds on cumulative leakage, similar to how differential privacy composes.

**Information Censorship Mechanism (ICM):** A concrete defense that probabilistically returns empty responses to bound expected leakage. The probability is calibrated based on worst-case information leakage. This demonstrates what a provably-safe defense looks like, even if it's not practical (high refusal rates kill utility).

The framework accomplishes something I think prior work was missing: it connects LLM safety to research on information theory, differential privacy, and statistical disclosure control. Concepts like composition theorems, privacy amplification, and rate-distortion tradeoffs become directly applicable.

## Other Observations / limitations

There are still some gaps I've noticed.

**The prior distribution problem:** Information-theoretic measures require knowing H(S), the entropy of sensitive information. But who defines what's "sensitive" and what's an adversary's prior knowledge? Different people have different threat models. The framework is mathematically rigorous but requires philosophical agreement on threat definitions.

**Adaptive adversaries:** The composition bounds might assume non-adaptive query strategies. Real adversaries use tools like PAIR (iterative refinement) or actively learn optimal query sequences. Does the framework provide guarantees against adversaries who optimize their queries based on previous responses and refusals?