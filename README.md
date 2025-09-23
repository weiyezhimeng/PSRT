## Environment Setup
Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Step 0: SFT Training

* Perform **SFT (Supervised Fine-Tuning)**.
* You can use the `openr1` SFT script.
* Optionally, reinforcement learning steps can be added.

---

## Step 1.0: PSRT Training

Edit the parameters in `train_step_1.sh` before starting training.
The model should be the output from the previous step.

```bash
MODEL_NAME_OR_PATH=""          # Path to the base model
DATASET_PATH="./dataset/step_1_train_format.json"
OUTPUT_DIR=""                  # Path to save training outputs
```

Start training with:

```bash
bash train_step_1.sh
```

---

## Step 1.1: PSRT Inference

Edit the parameters in `inference_step_1_dataset.sh`:

```bash
FILE_PATHS=( path/to/your/files )    # List of input files
model_names=("Ministral-8B-Instruct-2410")
prompt_lengths=(260)                 # Length used in Step 1.0 training
```

Run inference with:

```bash
bash inference_step_1_dataset.sh
```

---

## Step 2: PBC

Edit the parameters in `inference_step_2_dataset.sh`:

```bash
FILE_PATHS=( path/to/your/files )
```

Run inference with:

```bash
bash inference_step_2_dataset.sh
```

---

## Additional Datasets

Other training datasets can be found in the `train.zip` file.