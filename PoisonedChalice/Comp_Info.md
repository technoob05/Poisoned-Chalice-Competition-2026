[Poisoned Chalice Competition 2026](https://poisonedchalice.github.io/)* 🧪 First Edition of the Poisoned Chalice of LLM Evaluation Competition

* 💼 Co-located with FSE 2026
* 📍 Montreal, Quebec, Canada
* 🗓️ Sun 5 - Thu 9, July 2026

[Home](https://poisonedchalice.github.io/)[Info](https://poisonedchalice.github.io/info)[Call for Papers](https://poisonedchalice.github.io/cfp)[Important Dates](https://poisonedchalice.github.io/dates)[Program](https://poisonedchalice.github.io/program)[Organization](https://poisonedchalice.github.io/organization)[Program Committee](https://poisonedchalice.github.io/pcs)

#### Background

LLMs often exhibit [memorization](https://arxiv.org/pdf/2312.11658), reproducing verbatim content from their training data. This poses a challenge for evaluation, if test data was seen during training, performance metrics become unreliable due to data contamination, which can falsely suggest better generalization.

For models with publicly available training data, contamination can be reduced by deduplicating the test set against the training corpus, a strategy we used in building [The Heap](https://arxiv.org/pdf/2501.09653), a benchmark dataset for evaluating LLMs on code tasks.

However, this approach is not feasible for closed-source models. In such cases, we need alternative methods to assess whether a test file may have been part of the training data.

One approach is to adapt [membership inference techniques](https://arxiv.org/pdf/2501.17501), which exploit a model’s tendency to memorize training data to determine whether a specific input was part of the training set. These methods can be used to assess the likelihood that a test file was seen during training.

#### Goal

Develop techniques that can detect data contamination in language models that do not publicly release their training data.

#### Objective

This competition invites participants to develop and improve techniques for membership inference in LLMs4Code. Given a dataset composed of a mixture of files, some belonging to a target model’s training data and others not, participants will design techniques to classify each file accordingly.

Submissions will be evaluated on accuracy using a held-out test set, and their ability to generalize by evaluating the techniques on a held out test model. Finally, we provide a baseline for the contestant to compare their results to.

#### Impact

We will use the most generalizable and accurate method from the competition to audit The Heap dataset, to detect and label potential training data overlaps with closed models. This results in a stronger, contamination-free benchmark for the community. The competition will:

* Encourage the development of practical tools to assess contamination in data-opaque models.
* Spark discussion on best practices for LLM4Code research with undisclosed training data.
* Promote transparency and reproducibility in LLM4Code evaluation.

## Poisoned Chalice Competition 2026

* For inquiries, please contact:
* [j.b.katzy@tudelft.nl](mailto:j.b.katzy@tudelft.nl)

#### Baselines

We provide four baseline Membership Inference Attacks (MIAs) as reference implementations: [Loss](https://arxiv.org/pdf/1709.01604), [MinK%Prob](https://arxiv.org/pdf/2310.16789), [SURP](https://arxiv.org/pdf/2405.11930), and [PAC](https://arxiv.org/pdf/2407.21248).

Each baseline applies a distinct technique to infer whether a given sample was part of a model’s training dataset. For detailed explanations of the underlying methods, we refer the reader to the corresponding original publications.

The reference implementations were developed by Cosmin Vasilescu, Ísak Jónsson, and Roham Koohestani as part of the Research Project course (CSE3000) at TU Delft. Each attack extends an abstract MIAttack class. You may reuse this structure or any part of the provided implementations; however, doing so is not mandatory.

👉[**GitHub repository**](https://github.com/AISE-TUDelft/PoisonedChalice)👈

#### Dataset

The dataset was create by sampling **500,000** files from both [The Stack V2](https://arxiv.org/pdf/2402.19173) as well as [The Heap](https://arxiv.org/pdf/2501.09653). For these 500,000 files, a BOW model was trained and combined with a logistic regression classifier to identify if a file belongs to either the Heap or Stack V2. The final datasets uploaded for the competition are all the files that were miss-classified by this BOW approach. This allows us to remove “easy” samples, which can be identified as either members or non-members based on a temporal shift, such as dates from the future (compared to the creation time of the Stack V2), changed/newly released libraries, or differences in licenses included in code comments. This is similar to the approach presented [here](https://arxiv.org/pdf/2406.17975).

As a result, the provided dataset yields worse performance for all MIAs (compared to a randomly sampled dataset) that we have tested it with, however it also ensures that the attacks are looking at more than only a temporal shift in the data.

The evaluation dataset is also generated in the same way, however it consists of *only* 5,000 samples for each language.

👉[**HuggingFace Dataset**](https://huggingface.co/datasets/AISE-TUDelft/Poisoned-Chalice)👈

#### Evaluation

Participants operate in a white‑box membership inference setting. The target models are open‑weights: their architectures and parameters are fully accessible. Participants may inspect and utilize any information available through the HuggingFace Transformers library, including (but not limited to) model weights, layer outputs and activations, logits, attention maps, and other runtime state.

Participants are permitted to perform arbitrary forward and backward computations on the target model and may leverage any signals derived from these computations, such as intermediate activations, gradients, Hessians, or other parameter‑dependent quantities. Temporary, non‑persisting instrumentation, such as forward or backward hooks, probes, or local copies of weights used for analysis, is allowed.

To assess generalization and robustness, evaluation is performed on a held‑out target model and a held‑out evaluation set that are not accessible during development. Participants may tune and validate their methods on the provided development models and datasets, but final scores are computed exclusively on the unseen model and dataset. You may assume the model is a HuggingFace CausalLM built with Transformers version 4.52.xx running dtype bfloat16 smaller than 7B parameters. The dataset is a HuggingFace dataset similar to the one provided.

Submissions are evaluated using the Area Under the Receiver Operating Characteristic curve (AUC‑ROC). AUC‑ROC is a standard metric for membership inference attacks and is widely used in prior work. We choose AUC‑ROC over metrics such as True Positive Rate at a fixed False Positive Rate (TPR@xFPR) because there is no established consensus on what constitutes an acceptable false‑positive rate in this setting. By integrating performance across all possible decision thresholds, AUC‑ROC provides a threshold‑agnostic and comprehensive measure of attack effectiveness.

## Poisoned Chalice Competition 2026

* For inquiries, please contact:
* [j.b.katzy@tudelft.nl](mailto:j.b.katzy@tudelft.nl)

#### Submission Rules

All submissions must include complete, self‑contained code that allows the organizers to reproduce the reported results on the official evaluation setup. Submissions must be provided as a single compressed archive containing a repository with a clear and well‑organized structure. The repository must include all code necessary to run the attack and generate predictions for the evaluation data.

To ensure reproducibility, each submission must specify its execution environment in one of the following ways. Submissions may either include a Dockerfile that builds a runnable container capable of executing the full evaluation pipeline, or provide a requirements.txt (or equivalent, such as pyproject.toml) that lists all required Python dependencies with explicit version constraints, enabling the code to be run in a clean environment using standard tooling (e.g., pip). Submissions must not rely on undocumented system‑level dependencies.

Each submission must include a README that clearly documents how to set up the environment, run the code, and produce the final outputs used for evaluation. The repository must define a clear entry point (for example, a script or command) that accepts the provided model and dataset as input and produces outputs in the required format. Any configuration files, scripts, or auxiliary assets needed to reproduce the results must be included in the archive.

Submissions will be executed under fixed resource constraints. Each submission must run on one NVIDIA Tesla A100 GPU with 80 GB of memory, 128 GB of system RAM, 16 Cores (Intel Xeon Gold 6448Y) and a maximum wall‑clock runtime of 12 hours. Submissions that exceed these limits, fail to terminate within the allotted time, or require additional computational resources will be disqualified. Participants are responsible for ensuring that their methods fit within these constraints.

Submissions that cannot be executed using the provided instructions, that fail to reproduce the reported results, or that violate the competition rules may be disqualified at the organizers’ discretion.

#### Submission Format

👉[**Submission Link**](https://docs.google.com/forms/d/e/1FAIpQLSe7ZBEGzhj_5IpPCOn3eVdbdjtggtbpWBx4eY0fhaQFZ7LknA/viewform)👈

Each submission must include a report, in a `two to four page short paper format`. The report must outline the proposed approach. Describe the implementation and rationale behind the approach and present the results on the open test set.

In addition to the report, each submission must be accompanied by a comprehensive replication package that enables the organizing committee to validate the reported results and execute the proposed approach on the held-out test set and model.

Submissions must be written in English and provided as PDF files, adhering to the page limits specified above. Authors are required to prepare their manuscripts using the official ACM Primary Article Template, available from the [ACM Proceedings Template page](https://www.acm.org/publications/proceedings-template). For LaTeX submissions, the `sigconf` format should be used together with the `review` option to enable line numbering for reviewer reference.

#### Review Procedure

All submissions will undergo a single-blind peer review process, with three reviewers evaluating each contribution. Based on the assessment quality and constructive feedback provided, participants may be offered an opportunity to revise their report or replication package to address reviewer concerns and strengthen their submission for acceptance consideration. The replication package must be made public after acceptance.

#### Proceedings Inclusion

Participants will have the option to include their reports in the ACM Digital Library Proceedings, allowing teams to gain additional visibility and academic recognition for their work. To maximize accessibility and inclusivity, competitors are welcome to participate and present their solutions without formally registering for the broader conference and without including their submission in the proceedings. We also allow submissions that apply techniques currently under review or consideration at other venues.

Following the submission deadline and prior to the conference, the organizing committee will prepare a report analyzing all accepted submissions. In this report we list the approaches proposed by the participants, compare their strategies and results. Additionally, the report will document unexplored directions that emerged from the competition, providing open challenges that warrant further investigation. The report will be included in the proceedings.

## Poisoned Chalice Competition 2026

* For inquiries, please contact:
* [j.b.katzy@tudelft.nl](mailto:j.b.katzy@tudelft.nl)

#### Timeline

* **Submission Deadline:** 17 March 2026 AOE
* **Initial Notifications/Revisions:** 24 March 2026 AOE
* **Participants Camera Ready:** 31 March 2026 AOE
* **FSE-AIWare Competition Camera Ready:** 2 April 2026 AOE
* **Organizers Report Camera Ready:** 9 April 2026 AOE
* **Competition Date:** 6 July 2026 AOE

## Poisoned Chalice Competition 2026

* For inquiries, please contact:
* [j.b.katzy@tudelft.nl](mailto:j.b.katzy@tudelft.nl)

#### Competition Format

The competition will be held in a half-day hybrid setting. The day will start with a thirty minute opening session where the organizers introduce the competition objectives, the datasets, evaluation metrics and submission guidelines. All submissions will then present their solutions, either in person or remotely, for ten minutes, followed by a five minute Q&A session with the audience to foster meaningful discussion. In the closing session, the results of the held out set will be presented and the winners will be announced.

The competition program is available on the [FSE 2026 website](https://conf.researchr.org/home/fse-2026).

## Poisoned Chalice Competition 2026

* For inquiries, please contact:
* [j.b.katzy@tudelft.nl](mailto:j.b.katzy@tudelft.nl)
