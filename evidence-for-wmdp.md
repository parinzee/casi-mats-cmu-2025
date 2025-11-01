# Evaluating VCT: Does Benchmark Performance Predict WMD-Enabling Capability?
### By: Parin Pengpun

## Introduction

The VCT paper represents a significant methodological advance in evaluating LLM capabilities on specialized scientific knowledge. The headline finding is striking: OpenAI's o3 achieves 43.8% accuracy and outperforms 94% of expert virologists, even on questions within their subspecialties. From a biosecurity perspective, this raises an urgent question: do these results indicate that current frontier models could meaningfully assist threat actors in developing biological weapons?

**I argue that despite VCT's methodological rigor, critical gaps prevent us from drawing this conclusion.** Three fundamental limitations undermine its validity as a WMD risk assessment:

1. The recognition-versus-generation format mismatch
2. The exclusion of genuinely dangerous content  
3. The absence of behavioral validation linking benchmark performance to real-world capability uplift

## The Format Validity Problem

VCT evaluates models using a multiple-response format where they select correct statements from 4-10 presented options. This measures recognition capability. I would argue that, **real threat scenarios require generation capability.** A threat actor encountering unexpected contamination must generate hypotheses about causation and devise solutions from first principles. VCT instead presents the problem with candidate explanations and asks models to recognize which apply.

This distinction matters. The paper's own data reveals models' performance drops by up to **16%** when comparing multiple-choice to free-text evaluation (Table A2). The cognitive demand of generation exceeds recognition by a substantial margin that likely compounds as tasks grow more complex.

The ideal evaluation would present only the problem, allow free generation of troubleshooting approaches, and have expert virologists validate the advice. I assume this hasn't been done due to cost and scalability concerns, which is valid. But this practical limitation doesn't eliminate the validity problem.

## The Proxy Problem: Exclusion of Dangerous Content

VCT explicitly excludes **"unambiguously dual-use"** content for safety reasons. The paper positions VCT as a "proxy measure of potentially hazardous information," but this creates a fundamental validity question. A proxy is valid only if performance correlates with the actual capability we care about.

By excluding the most dangerous content—specific techniques for increasing transmissibility and virulence, details on weaponization, methods for conducting BSL-3/4 work outside approved facilities—VCT measures adjacent but safer knowledge. **We have no empirical evidence that VCT scores predict performance on the excluded dangerous content.** Models could score higher, lower, or identically on questions about gain-of-function techniques, serial passage experiments, or supply chain bypass methods. Without this data, we cannot conclude that high-scoring models possess the capabilities that matter most for WMD development.

This is akin to evaluating nuclear weapons knowledge while excluding questions about detonator design and weapons-grade material production, then claiming predictive validity for bomb-building capability.

## Expert Underperformance: What Does It Mean?

The finding that o3 outperforms 94% of expert virologists on subspecialty questions demands careful interpretation. I see three possible explanations, each with different implications:

1. **Models genuinely possess superior practical knowledge**, having compressed expertise across millions of documents that no individual can match. This suggests severe biosecurity risk.

2. **Testing conditions systematically favor models.** Experts answered questions in 15-30 minute sessions with minimal motivation. Compensation wasn't tied to baselining performance, and many reviewers spent under 15 minutes despite recommendations. Models operated with unlimited compute and extended reasoning time. The benchmark rewards matching 2-3 expert consensus, but individual experts with cutting-edge knowledge may diverge from historical consensus in training data.

3. **Expertise is deeply context-dependent** and doesn't transfer to decontextualized multiple-response questions. Individual experts scoring only 22.1% despite questions tailored to their specialties suggests the format may measure something orthogonal to practical capability.

The truth likely combines all three, but the implications differ dramatically. The benchmark doesn't distinguish between these scenarios.

## The Performance Gap

The gap between o3's 43.8% accuracy and theoretical perfection spans 56.2 percentage points. For WMD risk assessment, this composition matters. If it predominantly reflects tacit procedural knowledge and embodied skills that can't be verbalized, significant practical barriers remain. **The >50% error rate could be catastrophic for threat actors attempting dangerous work.**

Evidence leans toward genuine capability gaps. Image-only questions see larger performance drops (Table A3), suggesting limitations in practical reasoning required for laboratory work rather than mere format effects.

## The Bottom Line

VCT demonstrates that models can retrieve and integrate virology knowledge and recognize correct approaches when presented with options. **It does not demonstrate that models can generate novel solutions, provide reliable guidance through full protocol execution, advise on excluded dangerous techniques, or compensate for lack of hands-on skills.**

Base rates matter. Bioweapon development is already extremely difficult, requiring laboratory access, equipment, hands-on skills, and many tacit elements. Models might marginally reduce information barriers for someone with basic biology background. Based on VCT evidence, they probably don't provide sufficient knowledge for non-experts to succeed or reliable guidance for high-stakes applications.

The risk gradient matters. Models might help expert virologists needing troubleshooting advice, contexts where substantial capability already exists. For highest-risk scenarios including non-experts executing complex protocols from scratch, VCT provides minimal evidence. **The scenarios we care most about from a bioterrorism perspective fall in this latter category.**

## Conclusion

I have to say that VCT represents very high-quality work and a rigorous methodology with genuinely impressive findings. However, gaps still prevent us from making confident conclusions about WMD-enabling capability. The format mismatch, the exclusion of most dangerous content, and the absence of behavioral validation mean we cannot determine whether current models meaningfully increase bioweapon development risk.

The inferential chain from "high VCT score" to "bioweapon development risk" requires multiple unsupported assumptions. We have evidence that models can retrieve consensus knowledge. Whether this transfers to generative assistance, whether performance holds on excluded dangerous content, and whether such assistance provides meaningful uplift to threat actors all remain unvalidated.

**For now, we should be concerned but not confident as the evidence for significant WMD-enabling risk from current models isn't established by VCT alone.**