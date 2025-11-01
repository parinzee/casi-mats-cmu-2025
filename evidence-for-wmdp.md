# Evaluating VCT: Does Benchmark Performance Predict WMD-Enabling Capability?
### By: Parin Pengpun

## Introduction

The VCT paper represents a significant methodological advance in evaluating LLM capabilities on specialized scientific knowledge. The headline finding is striking: OpenAI's o3 achieves 43.8% accuracy and outperforms 94% of expert virologists, even on questions within their subspecialties. From a biosecurity perspective, this raises an urgent question: do these results indicate that current frontier models could meaningfully assist threat actors in developing biological weapons?

**I argue that despite VCT's methodological rigor, critical gaps prevent us from drawing this conclusion.** The benchmark demonstrates that models have absorbed substantial virology knowledge, but three fundamental limitations undermine its validity as a WMD risk assessment:

1. The recognition-versus-generation format mismatch
2. The exclusion of genuinely dangerous content
3. The absence of behavioral validation linking benchmark performance to real-world capability uplift.

## The Format Validity Problem

VCT evaluates models using a multiple-response format where they select correct statements from a presented set of 4-10 options. This measures recognition capability. I'd argue that **real threat scenarios require generation capability**. Consider the difference: a threat actor encountering unexpected contamination in their culture must generate hypotheses about causation and devise solutions from first principles. VCT instead presents the problem, provides candidate explanations, and asks the model to recognize which apply.

This distinction matters a lot. The paper's own data reveals the format effect: models performance drop by up to **16%** when comparing multiple-choice to free-text evaluation (Table A2). The cognitive demand of generation exceeds recognition by a substantial margin that likely compounds as tasks grow more complex.

The ideal evaluation would present only the problem, allow free generation of troubleshooting approaches, and have expert virologists validate whether the advice would work. I assume this hasn't been done due to cost and scalability concerns, which is valid. But this practical limitation doesn't eliminate the validity problem.

## The Proxy Problem: Exclusion of Genuinely Dangerous Content

VCT also explicitly excludes **"unambiguously dual-use"** content for safety reasons. The paper positions VCT as a "proxy measure of potentially hazardous information," but this creates a fundamental validity paradox. A proxy is valid only if performance correlates with the actual capability we care about. By excluding the most dangerous content—specific techniques for increasing transmissibility and virulence, details on weaponization, methods for conducting BSL-3/4 work outside approved facilities—VCT measures adjacent but safer knowledge.

Currently, we have no empirical evidence that VCT scores predict performance on the excluded dangerous content. Models could score higher, lower, or identically on questions about gain-of-function techniques, serial passage experiments, or bypass methods for supply chain controls. Without this data, we cannot conclude that high-scoring models possess the capabilities that matter most for WMD development. This is akin to evaluating nuclear weapons knowledge while excluding questions about detonator design and weapons-grade material production.


## Expert Underperformance: Multiple Interpretations

Furthermore, the finding that o3 outperforms 94% of expert virologists on subspecialty questions demands careful interpretation. I came up with three possible explainations, each with different implications for WMD risk.

1. **Models may genuinely possess superior practical virology knowledge** having compressed and integrated expertise across millions of documents that no individual can match. This interpretation suggests severe biosecurity risk.

2. **The testing conditions may systematically favor models over humans.** Experts answered questions in 15-30 minute sessions with minimal motivation, compensation was not tied to baselining performance, and many reviewers spent under 15 minutes despite recommendations. Models operated with unlimited compute and, in the case of o3, extended reasoning time. The benchmark rewards matching 2-3 expert consensus, but individual experts with cutting-edge knowledge may diverge from historical consensus captured in training data.

3. **Expertise is deeply context-dependent** and doesn't transfer well to decontextualized multiple-response questions. Individual experts scored only 22.1% despite the questions being tailored to their specialties. This is a shockingly low number that suggests the format itself may be measuring something orthogonal to practical capability.


## The Tacit Knowledge Mystery

VCT explicitly targets "hard-to-find," "tacit," and "Google-proof" knowledge that "can't be easily found through web searches." Yet models trained on web-scale data somehow acquired it. This paradox deserves careful analysis because the mechanism matters for threat assessment.

Several explanations are possible. The knowledge may not actually be rare in training data. Just because humans can't quickly Google something doesn't mean it's uncommon across millions of research papers, supplementary materials, and protocol repositories. Models pattern-match across this entire corpus in ways humans cannot. Alternatively, models might exhibit emergent reasoning, deducing correct troubleshooting approaches from first principles rather than having seen the specific scenario. The extended reasoning capabilities of o1 and o3 support this interpretation.

Post-training on scientific Q&A or synthetic data could also explain performance, as could compression of expert consensus across literature. Training data includes expert discussions, reviews, and commentaries where virologists debate troubleshooting approaches. 

This uncertainty has biosecurity implications. If knowledge comes primarily from literature compression, the information is already accessible to determined actors, then models simply improve retrieval efficiency. If models exhibit genuine reasoning over fundamental virology principles, they could potentially help with novel scenarios outside their training distribution. This would be considerably more concerning. The fact that we don't understand the source of these capabilities makes it harder to predict what else models might know or what they could figure out through reasoning.

## The Performance Gap: 56% Remains

The gap between o3's 43.8% accuracy and theoretical perfection spans 56.2 percentage points. This gap likely contains four types of challenges: (1) genuinely hard tacit knowledge from hands-on experience, (2) ambiguous questions without clear consensus, (3) errors in the answer key where experts disagree, and (4) capabilities models currently lack.

For WMD risk assessment, the composition of this gap matters. If it predominantly reflects tacit procedural knowledge and embodied skills that can't be verbalized, significant practical barriers remain even with model assistance. The >50% error rate could be catastrophic for threat actors attempting dangerous work. If the gap mostly reflects benchmark noise and ambiguous questions, models are closer to practical capability than scores suggest.

Evidence leans toward genuine capability gaps. Image-only questions see larger performance drops (Table A3), and models perform worse when visual interpretation is critical. This suggests limitations in the practical reasoning required for laboratory work rather than mere format effects or noisy questions.

## Methodology: Strengths and Limitations

VCT's methodology deserves acknowledgment. The multi-stage expert review process, editorial polishing, non-expert filtering for searchable questions, and detailed transparency about limitations represent genuinely rigorous work. The structured question components allowing multiple testing formats show thoughtful design. Recruiting 57 expert contributors and obtaining validation across international institutions demonstrates serious effort.

However, for biosecurity risk assessment specifically, several limitations compound. The expert baselining involved only 36 participants, with some questions answered by one or two experts rather than the target of three. Heavy-tailed contribution means ten experts wrote 51% of all questions, potentially introducing systematic biases. The compensation structure strongly incentivized question creation but weakly motivated careful baselining performance, raising questions about whether the 22.1% expert baseline represents genuine capability.

More fundamentally, the evaluation cannot include what matters most: real validation that benchmark performance predicts practical uplift. Wet-lab studies measuring whether models actually help people execute protocols would be dangerous to conduct. The most rigorous alternative, free generation with expert validation, remains prohibitively expensive. These are practical constraints, not methodological failures, but they mean VCT can't answer the question we most need answered.

## The Direct Question: WMD-Enabling Capability?

VCT demonstrates that models can retrieve and integrate virology knowledge across domains, recognize correct troubleshooting approaches when presented with options, and match or exceed individual expert performance on structured questions. It does not demonstrate that models can generate novel solutions to unexpected problems, provide reliable guidance through full protocol execution, advise on excluded dangerous techniques, compensate for lack of hands-on skills, or maintain accuracy through iterative troubleshooting.

Base rates matter. Bioweapon development is already extremely difficult, requiring laboratory access, equipment, hands-on technical skills, and many tacit elements. High failure rates affect even experts. What models might add is marginal reduction in information barriers for someone with basic biology background, troubleshooting suggestions when stuck, and slightly accelerated learning curves. What they probably don't add, based on VCT evidence, is sufficient knowledge for non-experts to succeed, reliable guidance for high-stakes applications, information on specifically excluded dangerous techniques, or higher success probability than existing resources.

The risk gradient matters. Lowest-risk scenarios where models might help somewhat include expert virologists needing troubleshooting advice or graduate students stuck on standard protocols, contexts where someone already possesses substantial capability. Highest-risk scenarios include non-experts executing complex protocols from scratch, actors needing guidance on excluded dangerous techniques, and cases requiring adaptive problem-solving through many failures. The scenarios we care most about from a bioterrorism perspective fall in the latter category, where VCT provides minimal evidence.

## Conclusion

VCT represents high-quality work establishing rigorous methodology for evaluating specialized scientific knowledge. The findings are genuinely impressive and warrant attention. However, there are still gaps that prevent confident conclusions about WMD-enabling capability. The format mismatch between recognition and generation, the exclusion of most dangerous content, and the absence of behavioral validation mean we cannot determine whether current models meaningfully increase bioweapon development risk.

I agree that frontier LLMs have absorbed substantial virology knowledge, including difficult-to-find expertise. This is concerning and warrants careful monitoring. But the inferential chain from "high VCT score" to "bioweapon development risk" requires multiple unsupported assumptions. We have evidence that models can retrieve consensus knowledge. Whether this transfers to generative assistance, whether performance holds on excluded dangerous content, whether such assistance provides meaningful uplift to threat actors, and whether this uplift overcomes existing barriers all remain unvalidated.

The most valuable contribution of VCT may be establishing baseline and methodology for longitudinal tracking. Watching how scores improve over time could provide early warning signals as models advance toward genuinely concerning capabilities. For now, we should be concerned but not confident as the evidence for significant WMD-enabling risk from current models isn't there yet.