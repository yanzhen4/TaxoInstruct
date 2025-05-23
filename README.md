# TaxoInstruct

Source code used for [TaxoInstruct](https://arxiv.org/abs/2402.13405) "A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion"

Our data and models are available at [this link](https://drive.google.com/drive/folders/1uzvK0jppBEni9B7Hy5OhZLQ9McbQym32?usp=drive_link)

We used the open-source framework [unsloth](https://github.com/unslothai/unsloth) to help us fine-tune LLMs efficiently. 

## 🚀 Scripts

This folder includes three shell scripts to run each task supported by **TaxoInstruct**:

| Script                          | Task                           | Description                                                                 |
|--------------------------------|--------------------------------|-----------------------------------------------------------------------------|
| `inference_SetExpan.sh`        | Entity Set Expansion           | Finds semantically similar sibling entities given seed examples.           |
| `inference_evaluate_TaxoExpan.sh` | Taxonomy Expansion           | Identifies appropriate parents for a new entity to insert into a taxonomy. |
| `inference_HiExpan.sh`         | Seed-Guided Taxonomy Construction | Grows a seed taxonomy by generating new entities and discovering parent-child edges. |

Each script is pre-configured with input/output paths and model checkpoints.

