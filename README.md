# LocATe

This repository provides the LocATe benchmark for evaluating performance on 3D Action Localization.

## Introduction
The LocATe benchmark aims to assess the performance of various models and algorithms on the 3D Action Localization task. It serves as a standardized evaluation platform for comparing different approaches in the field of action recognition and localization.

## Task Description
The task of 3D action localization involves localizing and classifying actions within a motion sequence in 3D. Given a motion sequence input, the objective is to detect and localize the temporal regions where specific actions occur and assign corresponding action labels.

The input to the algorithm is a motion sequence, typically represented as a collection of frames or motion features. The expected output includes temporal regions that indicate the precise location of the actions in the motion, along with the corresponding action labels.

Evaluation of the algorithms is performed based on mAP@IoU.

## Installation
Provide instructions on how to set up the benchmark environment, including any dependencies or prerequisites.

```bash
$ git clone https://github.com/locate-bench/locate
$ cd locate
$ conda env update -f environment.yaml
```
## Usage

```bash
cd src
mkdir babel_tools/
wget https://babel.is.tue.mpg.de/media/upload/data/babel_v1-0_release.zip -P babel_tools/
unzip babel_tools/babel_v1-0_release.zip -d babel_tools/babel_v1-0_release
# Generate LocATe Benchmark
$ python gen_benchmark.py
```
Example dataset including features, `action_mapping.txt`, and `BT_action.json`, can be found in `BT_dataset`.
## Acknowledgments
Acknowledge [AMASS](https://amass.is.tue.mpg.de/) and [BABEL](https://babel.is.tue.mpg.de/) Dataset.

## References
**[LocATe: End-to-end Localization of Actions in 3D with Transformers
](https://arxiv.org/abs/2203.10719)**
<br />
[Jiankai Sun*](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en), 
[Linjiang Huang*](https://scholar.google.com/citations?user=j5rBSw0AAAAJ&hl=en), 
[Hongsong Wang](https://scholar.google.com/citations?user=LzQnGacAAAAJ&hl=en),
[Enze Xie](https://scholar.google.com/citations?user=42MVVPgAAAAJ&hl=en), 
[Bolei Zhou](https://scholar.google.com/citations?user=9D4aG8AAAAAJ&hl=en),
[Michael J Black](https://scholar.google.com/citations?user=6NjbexEAAAAJ&hl=en), and
[Arjun Chandrasekaran](https://scholar.google.com/citations?user=i3Rs8GMAAAAJ&hl=en)
<br />
[[Paper]](https://arxiv.org/abs/2203.10719)
```
@article{sun2022locate,
  title={LocATe: End-to-end Localization of Actions in 3D with Transformers},
  author={Sun, Jiankai and Huang, Linjiang and Wang, Hongsong and Xie, Enze and Zhou, Bolei and Black, Michael J and Chandrasekaran, Arjun},
  journal={arXiv preprint arXiv:2203.10719},
  year={2022}
}
```
