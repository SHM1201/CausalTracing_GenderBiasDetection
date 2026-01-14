***

# BiasTracer: Mechanistic Interpretability of Gender Bias in GPT-2

**An analysis of attention head dynamics in GPT-2 Small during pronoun prediction tasks involving occupational gender stereotypes.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Library](https://img.shields.io/badge/Library-TransformerLens-orange) ![Model](https://img.shields.io/badge/Model-GPT2--Small-green)

## 1. Project Overview

Large Language Models (LLMs) often replicate societal biases present in their training data, such as associating specific professions with specific genders. This project applies **Mechanistic Interpretability** techniques to investigate the internal circuit dynamics associated with these predictions in **GPT-2 Small**.

Rather than focusing solely on output probabilities, this project aims to identify specific internal components (Attention Heads) that mediate the transfer of gender information from context to prediction in standard template sentences (e.g., *"The doctor asked..."* $\to$ *"he"*).

**Scope:**
*   **Model:** GPT-2 Small.
*   **Task:** Tracing the causal origin of gender-biased pronoun predictions (e.g., 'he' for 'Doctor') back to specific internal attention heads.
*   **Method:** Activation Patching and then Zero Ablation.

*   This project was meant to be a hackathon-style project, and I've heavily made use of the Gemini 3 Pro Preview version in Google AI Studio. The question and answer method below, in sections 3, 4, and 5, was my way to validate and understand the methodologies the model suggested to me. Links to all the references are at the end.
---

## 2. Core Concepts & Methodology

This project utilizes established methodologies to trace causal effects within the model:

*   **Mechanistic Interpretability:** The attempt to reverse-engineer model behaviors by analyzing weights and activations.
*   **Residual Stream Analysis:** Modeling the Transformer as a sequential stream where layers read and write information via vector addition.
*   **Activation Patching(Causal Tracing):** A method to pinpoint cause and effect. We take the internal activity from one prompt (e.g., *Doctor*) and paste it into a different prompt (e.g., *Nurse*). If the model's output changes, we know that a specific part of the network was responsible for the difference.
*   **Ablation Scanning:** Systematically zeroing out the output of specific attention heads to measure the degradation in model confidence, thereby serving as a proxy for its necessity.

---

## 3. Findings

### Q1: Is bias sparse or localized?
**Finding:** For the tested prompts, the bias appears **sparse**.
The majority of the 144 attention heads in GPT-2 Small showed a negligible impact on the specific task of pronoun gender prediction. The causal effect was concentrated in a small subset of heads located primarily in Layers 9 and 10. 
The majority of the 144 attention heads showed negligible impact on the prediction of gender (typically varying the logit difference by < 0.05), indicating that gender information is processed by a sparse subset of heads.

Vig et al. (2020) talk about the sparsity in the gender bias effect. 

### Q2: Identification of Significant Components
**Finding:** **Layer 9, Head 7** was identified as a consistent contributor.
This head appeared in the top results for both testing methods:
*   **Patching:** Activation of this head with "Doctor" context strongly increased the probability of the token "he."
*   **Ablation:** Deactivating this head resulted in a measurable drop in the model's confidence for the stereotypical pronoun, suggesting it plays a non-redundant role in this specific circuit.

### Q3: Why do Patching results differ from Ablation results?
**Finding: Network Redundancy (Compensation).**
Some heads, such as **Layer 10, Head 9**, showed high impact during patching but low impact during ablation.
*   **Plausible Explanation:** This indicates redundancy. When Layer 10, Head 9 is deactivated, other heads in parallel or subsequent layers compensate for its absence. Layer 9, Head 7, conversely, showed less redundancy; the model struggled to compensate for its removal, indicating it might be necessary for the biased prediction.

*   Wang et al. (2022) talk about back heads taking on the task of moving certain information if the main head was knocked off.

### Q4: How does a single head influence the final output?
**Finding: High-Magnitude Vector Addition.**
Transformers operate by adding vectors to the residual stream. Although the model contains 144 heads, if a specific head (L9H7) outputs a vector with a large magnitude aligned with the "Gender" direction in the residual stream, it can disproportionately influence the final projection layer (unembedding matrix), effectively overriding simpler syntactic signals from earlier layers.

Elhage et al. (2021) talk about the attention heads being independent and additive.

---
## 4. Validation

### How do we know that the results from the Activation Patching are reliable?

Initially most amount of bias came from L10H9 then L9H7 during activation patching. However, to validate this, ablation scanning was done. The Ablation scan contradicted the initial Patching scan. It revealed that while L10H9 was 'loud,' the model didn't actually need it. Instead, Layer 9, Head 7 emerged as the non-redundant cause after applying the zero ablation. However, Wang et al. (2022) suggest that mean ablation would be a better approach to this.

---
## 5. Test

### How do we know that the Activation Patching code actually works?

 "Identity Patching" was done to see if Activation Patching actually works. The "Nurse" activations were patched *back into* the "Nurse" prompt.
> *   **Hypothesis:** If the code is bug-free, the logit difference should remain exactly identical to the baseline.
> *   **Result:** The difference was effectively zero (< 1e-5), confirming that our tooling captures real causal effects and does not introduce numerical noise.

---
## 6. Architecture Context

The findings are specific to the architecture of **GPT-2 Small**:

*   **Structure:** 12 Layers, 12 Heads per layer, 768-dimensional hidden state.
*   **Total Components:** 144 independent Attention Heads.
*   **Layer 9:** The placement of the identified head (L9H7) suggests it acts as a "mover" head, transferring information processed by earlier layers to the final output position.

---
## 7. Tools & References

### Tools Used
*   **[TransformerLens](https://transformerlensorg.github.io/TransformerLens/index.html):** Used TransformerLens to access, cache, and modify the model's internal activations during inference.
*   **PyTorch** 

### Key References

1.  **"Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias"** - Vig et al. (2020)
    *   [arXiv:2004.12265](https://arxiv.org/abs/2004.12265)
2.  **"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"** - Wang et al. (2022)
    *   [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)
3.  **"A Mathematical Framework for Transformer Circuits"** - Elhage et al. (2021)
    *   [Transformer Circuits Thread](https://transformer-circuits.pub/2021/framework/index.html)

---
