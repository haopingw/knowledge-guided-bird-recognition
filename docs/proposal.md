# Can domain knowledge guide visual evidence used in bird recognition?

EECS 442: Computer Vision | Fall 2026 Project Proposal

Ziyi Wang, Haoping Wang, Jeffrey Huang, Yinuo Cao

## Motivation and research question

Bird species often differ in small details. We will test whether identification notes help Qwen3-VL recognize these differences and locate the relevant features. Inspired by classification using visual descriptions [1], we will study both recognition and evidence use in a frozen model that combines a visual encoder with a language decoder [2].

## Data and knowledge

CUB-200-2011 provides species labels, 15 part landmarks, and 312 binary attributes [3]. We will select 20 similar species using training data, reserve training images for validation, and evaluate on roughly 600 official test images. Species notes will summarize bird-guide descriptions [4], checked against training annotations. Test annotations will be used only for evaluation. Within each knowledge condition, every image receives notes for all candidate species; the true label never selects the supplied knowledge.

Figure 1. CUB training examples [3] share brown-gray plumage but differ in local features: (a) Chipping Sparrow, CUB #6743 — rust-colored crown, dark eye stripe; (b) White-crowned Sparrow, CUB #7706 — bold black-and-white crown stripes; (c) Song Sparrow, CUB #7527 — heavy streaking across the breast. Identification notes [4] direct inspection to the crown, eye stripe, or breast.

## Method

We will run Qwen3-VL-4B-Instruct without fine-tuning under five conditions: no added knowledge, a generic checklist, species identification notes, length-matched irrelevant notes, and notes with selected species-feature associations swapped. Candidate labels, resolution, output format, and generation limits stay fixed. Each response includes a species, coordinates for predefined parts, and visible attributes. Prompts will be selected on validation data and frozen before testing.

## Evaluation and expected findings

We will report mean per-class accuracy, attribute F1 on confidently annotated visible features, and landmark error normalized by the bird bounding-box diagonal, with paired bootstrap intervals. Masking diagnostic parts and equal-sized control regions on a fixed subset will test whether gains depend on relevant pixels. We expect useful notes to help with visible distinctions between similar species; misleading notes may expose reliance on text. If classification improves without better attribute or landmark predictions, we will examine whether notes mainly help interpretation. We will also measure latency and peak GPU memory.

## Feasibility and scope

A single RTX 5080 (16 GB; originally planned as RTX 5090) supports inference. Our four members will divide deployment, knowledge preparation, experiments, and evaluation. We will establish baselines, compare knowledge conditions, and then analyze masking tests and failures.

## References

[1] S. Menon and C. Vondrick. Visual Classification via Description from Large Language Models. ICLR, 2023. https://arxiv.org/abs/2210.07183
[2] Qwen Team. Qwen3-VL Technical Report. 2025. https://arxiv.org/abs/2511.21631
[3] C. Wah et al. The Caltech-UCSD Birds-200-2011 Dataset. Caltech, 2011. https://www.vision.caltech.edu/datasets/cub_200_2011/
[4] Cornell Lab of Ornithology. All About Birds: Bird Guide. https://www.allaboutbirds.org/guide/
