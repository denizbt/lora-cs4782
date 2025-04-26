# Datasets Used For Results
The data used in this implementation of LoRA is the datasets associated with tasks in the GLUE benchmark.

The dataset for each task in GLUE can be downloaded using HuggingFace's Datasets package. First ensure, you have the datasets package installed in your environment (if you are using an environment derived from this repository's `requirements.txt`, you should already have the requisite packages installed). Otherwise, run
```
pip install datasets
```

Then, in your Python script, add
```
from datasets import load_dataset

target_task = "mnli" # MNLI task chosen as example
task_dataset = load_dataset("glue", target_task) 
```