***

# BiasTracer: Mechanistic Interpretability of Gender Bias in GPT-2

**An analysis of attention head dynamics in GPT-2 Small during pronoun prediction tasks involving occupational gender stereotypes.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Library](https://img.shields.io/badge/Library-TransformerLens-orange) ![Model](https://img.shields.io/badge/Model-GPT2--Small-green)

## 1. Project Overview

Large Language Models (LLMs) often replicate societal biases present in their training data, such as associating specific professions with specific genders. This project applies **Mechanistic Interpretability** techniques to investigate the internal circuit dynamics associated with these predictions in **GPT-2 Small**.

Rather than focusing solely on output probabilities, this project aims to identify specific internal components (Attention Heads) that mediate the transfer of gender information from context to prediction in standard template sentences (e.g., *"The doctor asked..."* $\to$ *"he"*).

**Scope:**
*   **Model:** GPT-2 Small.
*   **Method:** Activation Patching and Mean Ablation.
*   **Task:** Pronoun resolution in occupational contexts.

---

## 2. Core Concepts & Methodology

This project utilizes established methodologies to trace causal effects within the model:

*   **Mechanistic Interpretability:** The attempt to reverse-engineer model behaviors by analyzing weights and activations.
*   **Residual Stream Analysis:** Modeling the Transformer as a sequential stream where layers read and write information via vector addition.
*   **Activation Patching(Causal Tracing):** A method to pinpoint cause and effect. We take the internal activity from one prompt (e.g., *Doctor*) and paste it into a different prompt (e.g., *Nurse*). If the model's output changes, we know that specific part of the network was responsible for the difference.
*   **Ablation Scanning:** Systematically zeroing out the output of specific attention heads to measure the degradation in model confidence, thereby serving as a proxy for its necessity.

---

## 3. Findings

### Q1: Is bias sparse or localized?
**Finding:** For the tested prompts, the bias appears **sparse**.
The majority of the 144 attention heads in GPT-2 Small showed negligible impact on the specific task of pronoun gender prediction. The causal effect was concentrated in a small subset of heads located primarily in Layers 9 and 10. 
The majority of the 144 attention heads showed negligible impact **on the differential prediction of gender** (typically varying the logit difference by < 0.05), indicating that gender information is processed by a sparse subset of heads.

### Q2: Identification of Significant Components
**Finding:** **Layer 9, Head 7** was identified as a consistent contributor.
This head appeared in the top results for both testing methods:
*   **Patching:** Activation of this head with "Doctor" context strongly increased the probability of the token "he."
*   **Ablation:** Deactivating this head resulted in a measurable drop in the model's confidence for the stereotypical pronoun, suggesting it plays a non-redundant role in this specific circuit.

### Q3: Why do Patching results differ from Ablation results?
**Finding: Network Redundancy (Compensation).**
Some heads, such as **Layer 10, Head 9**, showed high impact during patching but low impact during ablation.
*   **Explanation:** This indicates redundancy. When Layer 10, Head 9 is deactivated, other heads in parallel or subsequent layers compensate for its absence. Layer 9, Head 7, conversely, showed less redundancy; the model struggled to compensate for its removal, indicating it is structurally necessary for the biased prediction.

### Q4: How does a single head influence the final output?
**Finding: High-Magnitude Vector Addition.**
Transformers operate by adding vectors to the residual stream. Although the model contains 144 heads, if a specific head (L9H7) outputs a vector with a large magnitude aligned with the "Gender" direction in the residual stream, it can disproportionately influence the final projection layer (unembedding matrix), effectively overriding simpler syntactic signals from earlier layers.

---

## 4. Architecture Context

The findings are specific to the architecture of **GPT-2 Small**:

*   **Structure:** 12 Layers, 12 Heads per layer, 768-dimensional hidden state.
*   **Total Components:** 144 independent Attention Heads.
*   **Layer 9:** The placement of the identified head (L9H7) suggests it acts as a "mover" head, transferring information processed by earlier layers to the final output position.

---

## 5. Tools & References

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
