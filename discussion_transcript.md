diff --git a/\\wsl.localhost\Ubuntu\home\guillaumeminet\LISAI\docs/discussion_transcript.md b/\\wsl.localhost\Ubuntu\home\guillaumeminet\LISAI\docs/discussion_transcript.md
new file mode 100644
--- /dev/null
+++ b/\\wsl.localhost\Ubuntu\home\guillaumeminet\LISAI\docs/discussion_transcript.md
@@ -0,0 +1,317 @@
+## User
+
+# Context from my IDE setup:
+
+## Active file: configs/training/hdn.yml
+
+## Open tabs:
+- hdn.yml: configs/training/hdn.yml
+- preprocess.yml: configs/preprocess/preprocess.yml
+- config_train.yaml: e:\dl_monalisa\Models\Gag\Upsamp\CL5_Upsamp2\config_train.yaml
+- continue_training.yml: configs/training/continue_training.yml
+- fast_upsamp.yml: configs/training/fast_upsamp.yml
+
+## My request for Codex:
+I need your advice. Currently my project settings define a data/model architecture where datasets live in
+root/Data/
+and models in
+root/Models/ with each model being in a subfolder with the trained dataset
+
+I think it was good during developpment to have a clear separation between datasets and models
+
+## User
+
+# Context from my IDE setup:
+
+## Active file: configs/training/hdn.yml
+
+## Open tabs:
+- hdn.yml: configs/training/hdn.yml
+- preprocess.yml: configs/preprocess/preprocess.yml
+- config_train.yaml: e:\dl_monalisa\Models\Gag\Upsamp\CL5_Upsamp2\config_train.yaml
+- continue_training.yml: configs/training/continue_training.yml
+- fast_upsamp.yml: configs/training/fast_upsamp.yml
+
+## My request for Codex:
+But now that i'm transitionning to a more "developmment finished; using the code routinely" mode, i realize its not super practical to jump between the datasets and the models. 
+Also, it's not easy to see which models are available to use on actual data (not training data) 
+So i'm considering changing for
+root/datasets/*datasetname*/
+       -models
+        -data
+and having another folder for actual data i use the models on, such 
+root/*actual_data* (but i need your input on a better nme here
+
+What do you think, what would you recommend for a good organization of datasets vs models vs actual data to use on
+
+On the same note of making it easier to see which models are available to use on actual data, I initially starte the dataset registry which sits in 
+root/data/registry.yml
+
+As a way to keep track of all the training datasets, registring them in the preprocess step.
+
+But, first of all, i'm not really using it right now, it's not really useful.
+So i'm looking for a better way of using it.
+
+Second of all, I'd like to have a similar registry for trained models, to easily see which models are trained on which dataset. One solution could be to couple it with dataset registry, so we could see easily for each dataset the dataset characteristic and the trained models associated.
+Another thing i'm considering is having a feature to see which models are "usable", ie:
+there are traind models that have been done for tetsing; benchmarking, etc
+But once you go to models you actually want to use, you might have only a few, and it would be nice if they stand-out. So something like "usable_models", either as a separate registry or a separate category; that we could then use more easily. For example we could give it an alias name and then easily do
+lisai apply alias name... 
+
+And lisai apply could maybe be restrictive to only registred models as "usable" in the future.
+
+I'm looking for a general opinion and advice on how to achieve that kind of goals without fully changing the codebase either
+
+## Codex
+
+I would move to a dataset-centric training tree, but I would keep routine inference/apply data separate from training datasets.
+
+```text
+root/
+  datasets/
+    Gag/
+      dump/
+      preprocess/
+      models/
+        HDN/
+          HDN_Gag_KL07_01/
+        Upsamp/
+          ...
+  inference/
+    case_001/
+      input/
+      output/
+```
+
+If your non-training work is organized by experiment or collaboration, `cases/` or `projects/` is even better than `actual_data/`. My default recommendation would be `inference/`; `actual_data` becomes vague once some training data is also “actual”.
+
+This is a pretty safe direction for LISAI because the filesystem layout is already centralized in [configs/project_config.yml](/home/guillaumeminet/LISAI/configs/project_config.yml#L10) and [src/lisai/infra/paths/paths.py](/home/guillaumeminet/LISAI/src/lisai/infra/paths/paths.py#L19). Training already thinks in terms of `dataset_name`, `data_subfolder`, and `models_subfolder` rather than raw paths, for example in [configs/training/hdn.yml](/home/guillaumeminet/LISAI/configs/training/hdn.yml#L14). Also, `lisai apply` already accepts an arbitrary input path and, by default, saves next to that input, so your routine apply data does not need to live under the training dataset tree at all: [src/lisai/evaluation/cli.py](/home/guillaumeminet/LISAI/src/lisai/evaluation/cli.py#L62) and [src/lisai/evaluation/run_apply_model.py](/home/guillaumeminet/LISAI/src/lisai/evaluation/run_apply_model.py#L120).
+
+For registries, I would not make the dataset registry carry your “usable model” workflow. Preprocess rewrites that file automatically in [src/lisai/data/preprocess/run_preprocess.py](/home/guillaumeminet/LISAI/src/lisai/data/preprocess/run_preprocess.py#L136), and saving uses `yaml.safe_dump` in [src/lisai/config/io/yaml.py](/home/guillaumeminet/LISAI/src/lisai/config/io/yaml.py#L20), so it is not a great home for curated operational metadata. I’d keep:
+- `dataset_registry.yml`: auto-updated technical dataset facts
+- `model_registry.yml` or `usable_models.yml`: curated aliases for models you actually want to use
+
+Example:
+
+```yaml
+gag_hdn:
+  run: Gag/HDN/HDN_Gag_KL07_01
+  dataset: Gag
+  status: usable
+  task: denoise_timelapse
+```
+
+Then support both `lisai apply gag_hdn ...` and `lisai apply Gag/HDN/HDN_Gag_KL07_01 ...`. I would not fully restrict `apply` to registered models; aliases should be the convenient path, not the only path.
+
+One concrete reason the current dataset registry may feel underused: preprocess writes `format`, but the resolved data path checks `dataset_info["data_format"]`, so that metadata does not seem to round-trip cleanly today: [src/lisai/data/preprocess/core/dataset_registry.py](/home/guillaumeminet/LISAI/src/lisai/data/preprocess/core/dataset_registry.py#L69) and [src/lisai/config/models/training/data.py](/home/guillaumeminet/LISAI/src/lisai/config/models/training/data.py#L411).
+
+If you want the lowest-risk path without reshaping the whole codebase, I’d do this in order:
+1. Change only the path templates so runs live under `datasets/<dataset>/models/...`.
+2. Fix the registry key mismatch.
+3. Add a separate curated model alias registry.
+4. Extend `apply` so a slash-free token is treated as an alias.
+
+If you want, I can turn that into a concrete migration plan with the exact files and the smallest code changes first.
+
+## User
+
+# Context from my IDE setup:
+
+## Active file: configs/training/hdn.yml
+
+## Open tabs:
+- hdn.yml: configs/training/hdn.yml
+- experiment.schema.json: configs/schema/experiment.schema.json
+- cli.py: src/lisai/training/cli.py
+- cli.py: src/lisai/cli.py
+- preprocess.yml: configs/preprocess/preprocess.yml
+
+## My request for Codex:
+how would you register the models in theh curated model alis registry? by hand?
+
+## Codex
+
+Not fully by hand.
+
+I’d split it like this:
+
+- Manual: alias, `status: usable`, maybe a short note like “best for noisy timelapse”.
+- Automatic: dataset, architecture, run ref, checkpoint defaults, normalization-related metadata.
+
+That split fits LISAI well because every saved run already contains a `config_train.yaml` written at save time in [save_training_config]( /home/guillaumeminet/LISAI/src/lisai/training/setup/run_dir.py#L92 ), and evaluation already knows how to reload that metadata through [load_saved_run]( /home/guillaumeminet/LISAI/src/lisai/evaluation/saved_run.py#L139 ). Also, the existing [MODEL_REGISTRY]( /home/guillaumeminet/LISAI/src/lisai/models/registry.py#L3 ) is only for model classes like `lvae` or `unet`, not for trained runs.
+
+So my recommendation would be:
+
+1. Start with a small hand-edited curated file, probably `configs/model_aliases.yml`.
+2. Keep it tiny and human-owned.
+3. Add a helper command later that fills most fields from the saved run and only asks for alias/status.
+
+Example curated file:
+
+```yaml
+gag_hdn:
+  run: Gag/HDN/HDN_Gag_KL07_01
+  status: usable
+  task: denoise_timelapse
+  note: default model for routine Gag inference
+
+gag_hdn_benchmark:
+  run: Gag/HDN/HDN_Gag_KL05_03
+  status: benchmark
+  task: denoise_timelapse
+```
+
+If you do it by hand at first, I would only require `run` and `status`. Everything else can stay optional.
+
+Best long-term workflow:
+
+- `lisai train ...`
+- inspect results
+- if the run is worth keeping, “promote” it
+- promotion writes one alias entry
+- `lisai apply gag_hdn /path/to/data`
+
+So yes: hand-curated for the promotion decision, but not hand-maintained for all technical metadata. That keeps the important human judgment manual without making the process tedious.
+
+If you want, I can sketch the smallest possible `model_aliases.yml` schema and the simplest `lisai apply <alias>` resolution path.
+
+## User
+
+# Context from my IDE setup:
+
+## Active file: configs/training/hdn.yml
+
+## Open tabs:
+- hdn.yml: configs/training/hdn.yml
+- experiment.schema.json: configs/schema/experiment.schema.json
+- cli.py: src/lisai/training/cli.py
+- cli.py: src/lisai/cli.py
+- preprocess.yml: configs/preprocess/preprocess.yml
+
+## My request for Codex:
+But what about having a more general dataset registry, something automatic like for each dataset you have
+Gag:
+     *run_name_1*:
+              mode: ...
+              subfolder: "HDN"
+              start-stop: datetime start - datetime stop
+              status: .... # finished / ongoing / stopped / failed
+              n_epochs: ... 
+               # maybe other useful info?
+
+or maybe even better, organized per subfolder inside each dataset so
+
+Gag:
+      HDN:
+             *run_name*: 
+                          ...
+       Upsamp:
+               *run_name*:
+                            ...
+
+The goal would be quickly be able to see which models have been trained, etc
+
+And we could have a training parameter ; add_to_registry (bool) to automatically add it or not. We would then update its status depending on what's going on during the training
+
+I'm not sure how to relate this to a the curated model registry with aliases, but this could be a first step to have a quick and easy overview of trained models, ongoing training, etc.      
+
+## Codex
+
+Yes, the overview you want makes a lot of sense. I just would not put it into the current dataset registry as the primary source of truth.
+
+Right now that registry is dataset-oriented and is loaded as `dataset_info` during training and evaluation in [prepare_data]( /home/guillaumeminet/LISAI/src/lisai/training/setup/data.py#L66 ) and [resolve_dataset_info]( /home/guillaumeminet/LISAI/src/lisai/evaluation/data.py#L22 ). It is also rewritten by preprocess in [run_preprocess]( /home/guillaumeminet/LISAI/src/lisai/data/preprocess/run_preprocess.py#L136 ). Live training status has a different lifecycle, so I would separate it.
+
+What I would recommend is a 3-layer model:
+
+- Dataset registry: dataset facts only.
+- Run metadata registry: automatic, for all training runs.
+- Curated alias registry: manual, only for promoted usable models.
+
+For the automatic layer, I actually prefer this as the source of truth:
+
+- one small metadata file inside each run directory
+- then optionally a generated grouped view by dataset/subfolder
+
+That fits the current code very naturally, because training already revolves around a canonical run directory in [prepare_run_dir]( /home/guillaumeminet/LISAI/src/lisai/training/setup/run_dir.py#L34 ), saves `config_train.yaml` there in [save_training_config]( /home/guillaumeminet/LISAI/src/lisai/training/setup/run_dir.py#L92 ), and already writes other per-run artifacts like `loss.txt` through [CheckpointManager]( /home/guillaumeminet/LISAI/src/lisai/training/checkpointing/manager.py#L12 ). It also already has clear stop states in [TrainingOutcome]( /home/guillaumeminet/LISAI/src/lisai/training/trainers/base.py#L20 ) and final exception handling in [run_training]( /home/guillaumeminet/LISAI/src/lisai/training/run_training.py#L111 ).
+
+So I’d make each run have something like `run_status.yml`:
+
+```yaml
+status: running
+mode: train
+dataset_name: Gag
+models_subfolder: HDN
+data_subfolder: preprocess/recon_noisy
+architecture: lvae
+started_at: 2026-03-19T15:12:03
+stopped_at: null
+last_completed_epoch: 17
+target_epochs: 100
+best_val_loss: 0.0142
+```
+
+Then, if you want the quick overview, you can expose it as a grouped view like:
+
+```yaml
+Gag:
+  HDN:
+    HDN_Gag_KL07_01:
+      status: completed
+      mode: train
+      last_completed_epoch: 99
+      target_epochs: 100
+```
+
+That grouped view is good, but I would make it derived, not the thing training writes directly. The reason is concurrency: a single shared YAML being updated by multiple trainings is much more fragile than one metadata file per run.
+
+For the config flag, I would use `register_run: true`, not `add_to_registry`. If you want the smallest change surface, put it under `saving`. If you want cleaner semantics, make a tiny new `tracking` section. For a first version, `saving.register_run: true` is perfectly fine.
+
+The relation to the curated alias registry is simple:
+
+- automatic run metadata answers: “what exists, what is running, what finished, what failed?”
+- curated alias registry answers: “which of those runs should people actually use?”
+
+So the alias registry can just point to one run from the automatic layer:
+
+```yaml
+gag_hdn:
+  run: Gag/HDN/HDN_Gag_KL07_01
+  status: usable
+```
+
+If you want, I can sketch the minimal implementation plan for this with the exact hooks I’d use:
+- create the run metadata at runtime initialization
+- update it at each epoch end
+- finalize it on `completed` / `early_stopped` / `interrupted` / `failed`
+
+## User
+
+# Context from my IDE setup:
+
+## Active file: configs/preprocess/preprocess.yml
+
+## Open tabs:
+- preprocess.yml: configs/preprocess/preprocess.yml
+- run_preprocess.py: src/lisai/data/preprocess/run_preprocess.py
+- snr.py: src/lisai/data/preprocess/transformations/snr.py
+- spatial.py: src/lisai/data/preprocess/transformations/spatial.py
+- temporal.py: src/lisai/data/preprocess/transformations/temporal.py
