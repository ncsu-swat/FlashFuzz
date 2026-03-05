#set page(paper: "us-letter", margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)

#align(center)[
  #text(size: 14pt, weight: "bold")[Artifact Abstract: How Effective Is Coverage-Guided Fuzzing to Test Deep Learning Library APIs?]

  #v(0.5em)

  #text(size: 11pt)[
    Feiran (Alex) Qin#super[1], M M Abid Naziri#super[1], Hengyu Ai#super[2], Saikat Dutta#super[3], Marcelo d'Amorim#super[1] \
    #text(size: 9pt)[#super[1]North Carolina State University #h(1em) #super[2]ShanghaiTech University #h(1em) #super[3]Cornell University]
  ]

  #v(1em)
]

= Paper Title

How Effective Is Coverage-Guided Fuzzing to Test Deep Learning Library APIs?

= Artifact Purpose

This artifact accompanies the above paper, accepted to the ICST 2026 Research Track. It contains the complete implementation of #smallcaps[FlashFuzz], a framework that employs coverage-guided fuzzing (CGF) to test Deep Learning (DL) library APIs at scale. #smallcaps[FlashFuzz] leverages Large Language Models (LLMs) to automatically synthesize API-level test harnesses for PyTorch and TensorFlow, enabling effective CGF for DL libraries.

The artifact includes:
- Source code for #smallcaps[FlashFuzz] (experiment runner, Docker orchestration, test harness generation).
- Pre-generated test harnesses for 1,271 PyTorch and 786 TensorFlow APIs.
- Dockerfiles that build PyTorch and TensorFlow from source with coverage instrumentation and fuzzer support.
- Baseline configurations for ACETest, PathFinder, and TitanFuzz.
- Ablation study harness variants (with/without helper functions, with/without documentation).
- Jupyter notebooks and post-processing scripts to reproduce all figures and tables in the paper.

= Badges Claimed

*Artifact Available.* The artifact is publicly available at #link("https://github.com/ncsu-swat/FlashFuzz/tree/artifact-evaluation") under the MIT license, ensuring long-term public availability.

*Artifact Reviewed.* We claim the Reviewed badge on all four criteria:
- *Documented:* The repository includes a detailed `README.md` with step-by-step instructions for setup, a kick-the-tires quick test (~30 min), and full experiment reproduction.
- *Consistent:* The artifact directly produces the results reported in the paper (coverage comparisons, ablation study, validity analysis, and bug detection).
- *Complete:* All components needed to reproduce the paper's experiments are included: source code, test harnesses, Docker images, baseline configurations, API lists, and plotting scripts.
- *Exercisable:* Pre-built Docker images are available on Docker Hub. The kick-the-tires evaluation can be completed in ~30 minutes. All scripts and notebooks can be executed to reproduce the paper's results.

= Technology Skills

Reviewers should be familiar with:
- *Docker* — building and running containers.
- *Python* — running Python scripts from the command line.
- *Jupyter Notebooks* — executing notebook cells to reproduce plots.
- *Linux command line* — basic shell operations.

No knowledge of fuzzing internals, C++ test harness code, or DL library internals is required to run the artifact.

= Access and Environment

*Repository:* #link("https://github.com/ncsu-swat/FlashFuzz/tree/artifact-evaluation")

*Pre-built Docker images:* Available on GitHub Container Registry under `ghcr.io/ncsu-swat/flashfuzz:*`. Pulling images avoids the multi-hour build step.

*Hardware requirements:*
#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: left,
  table.header([*Resource*], [*Minimum*], [*Recommended*]),
  [CPU cores], [8], [50+ (for parallel fuzzing)],
  [RAM], [16 GB], [64 GB+],
  [Disk], [50 GB free], [200 GB+],
  [GPU], [Not required], [Optional (NVIDIA + Container Toolkit)],
)

*Operating system:* Linux (tested on Ubuntu 22.04). Docker must be installed and accessible without `sudo`.

*Software dependencies:* Python 3.10+, `tqdm`, `bs4`, `regex`. A `pixi.toml` is provided for one-command environment setup via #link("https://pixi.sh")[Pixi].

*Setup time:* ~10 minutes with pre-built Docker images (pull + pixi install). ~2--4 hours if building Docker images from source.
