# TaxoInstruct

Official Repo for [TaxoInstruct](https://arxiv.org/abs/2402.13405) "A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion", which was accepted at ***ACL 2025 (Findings)*** and ***AI4Research @ AAAI 2025*** (Oral Presentation). You can find the slide for the presentation at [this link](https://drive.google.com/drive/folders/1wzfGhnd494RoEx0T6ubcwcFP9A_35jfI?usp=sharing). 

Our data and models are available at [this link](https://drive.google.com/drive/folders/1uzvK0jppBEni9B7Hy5OhZLQ9McbQym32?usp=drive_link).

We used the open-source framework [unsloth](https://github.com/unslothai/unsloth) to fine-tune LLMs efficiently, please follow unsloth's documentation to set up the environment. 



## Scripts

This folder includes three shell scripts to run the three taxonomy construction tasks supported by **TaxoInstruct**:

| Script                          | Task                          | Available Datasets
|--------------------------------|--------------------------------|--------------------------------|
| `inference_SetExpan.sh`        | Entity Set Expansion           | APR, WIKI
| `inference_evaluate_TaxoExpan.sh` | Taxonomy Expansion           | Environment, Science
| `inference_HiExpan.sh`         | Seed-Guided Taxonomy Construction | CVD, DBLP

Each script is pre-configured with input/output paths and model checkpoints. Before running the script, put the folder (i.e., ```datasets/``` and ```model/```) under the repository main folder ```./.```



