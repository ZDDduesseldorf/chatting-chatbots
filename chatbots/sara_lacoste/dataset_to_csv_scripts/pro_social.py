from datasets import load_dataset

split = "train"

dataset = load_dataset("allenai/prosocial-dialog", split=split)

dataset = dataset.rename_column("context", "Input")
dataset = dataset.rename_column("response", "Output")
dataset = dataset.remove_columns([
    "rots", "safety_label", "safety_annotations",
    "safety_annotation_reasons", "source", "etc",
    "dialogue_id", "response_id", "episode_done"
])
pro_social = dataset.to_pandas()
pro_social.to_csv(rf"data/prosocial_{split}.csv", index=False, sep=';')
