## For target inference

If there exist new interested compound with perturbation transcriptomics to conduct target inference, plz follow this step.

### Step1:Processing your own transcriptomics data

This step depends entirely on the format of your data. You can process it using your own differential expression gene (DEGs) analysis code. Generally, two strategies are recommended:

1. Use the "top strategy" to identify DEGs, as described in our paper.
2. Alternatively, use a standard DEG pipeline, such as Limma, etc.

In either case, the final data format should be as following triples:

```
<Interested_compound_name  Downregulates  mRNA:ATP1B1>
# Note: aligning gene name using our map_file
```

### Step2:Preparing training file

We need to prepare four files:**"cause.txt", "process.txt", "effect.txt", and "test.txt"**.

The simplest approach I recommend is to directly use the files from the directory "../processed_data/target_inference_2/". The only part that needs modification is **adding the triplets identified in the first step to the "effect.txt" file**. 

Then, place "cause/effect/test"  into a new folder, such as "../processed_data/<name>/", where <name> can be any name you choose to name the task.

## Step3:Training stage

Using following cmd to train the PertKGE by replacing  **<name>**:

```
$ python train_pertkge.py --cause_file "../processed_data/<name>/cause.txt"\
                          --process_file "../processed_data/knowledge_graph/process.txt"\
                          --effect_file "../processed_data/<name>/effect.txt"\
                          --test_file "../processed_data/<name>/test.txt"\
                          --h_dim 300\
                          --margin 1.0\
                          --lr 1e-4\
                          --wd 1e-5\
                          --n_neg 100\
                          --mode 'reproduce'\
                          --batch_size 2048\
                          --patients 5\
                          --warm_up 10\
                          --processed_data_file "../processed_data/<name>/"\
                          --save_model_path "../best_model/<name>/"\
                          --task "target_inference"\
                          --run_name "new_target_inference"
```

### Step4:Inference stage

Please follow the steps sequentially as outlined in the `target_inference.ipynb`. Adjust the paths in the user-defined cells accordingly. 

```
'''
This section is user defined !!!
'''
h_dim = 300

data_path = "../processed_data/<name>/"
save_model_path = '../best_model/<name>/'
output_path = "../results/<name>/"

ent_list = ['Interested_compound_name']  # The `ent_list` refers to the list of compounds you wish to infer, which should correspond to the compound names you processed in the step 1.
```

### Another notation

In target inference, we provide two metrics: "ti_score" and "confidence".

- **"ti_score"** is the score directly predicted by the model.
- **"confidence"** represents how many other compounds were ranked lower than this compound for the target. This metric can help filter out potential false positives caused by representational bias.



## ligand virtual screening

The simplest approach is to directly use the model weights we provided for virtual screening.
