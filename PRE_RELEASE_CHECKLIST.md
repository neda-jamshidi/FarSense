# Release checklist (maintainers)

- [ ] Confirm the public paper title, author list, venue/year and add a formal citation to `README.md`.
- [ ] Choose/confirm the code and dataset licenses; do **not** add a license without all rights-holders' approval.
- [ ] Confirm which dataset split and model checkpoints were used for the reported paper results. These are **not contained** in the supplied archives.
- [ ] If available, link to public base models, LoRA adapters and the HF train/test dataset; note required access permissions and GPU memory.
- [ ] Review the mismatch between the `true_false_2` prompt's example key (`true_false_1_answer_predict`) and its actual reference key (`true_false_2_answer_predict`). The lightweight evaluator accepts both for that task.
- [ ] Verify the historic training/inference scripts on a compatible GPU/Python/Transformers/TRL/PEFT environment; they were reorganized, not executed end to end here.
- [ ] Verify the original `eval3_original.py` metrics/model choices are the same as those used in the paper before calling them reproducibility results. It uses English DeBERTa for BERTScore even though FarSense is Persian.
- [ ] When publishing, avoid including generated outputs containing any unpublished/private model data.
- [ ] If necessary, add sample model predictions and a command that reproduces **each published table** on the released test split.
