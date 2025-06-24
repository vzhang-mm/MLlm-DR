
pretraining LLMs [LLaMA-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct). 

## 📁 Project Structure

* `main.py`: Main training script
* `result_evaluation.py`: Evaluation script (outputs metrics like MAE, RMSE, F1)
* `./result`: Folder to save experiment outputs
* `./data/CMDC`: Directory for the CMDC dataset (you must prepare this beforehand)

```
├── data/
│   ├── CMDC/
│   └── E-DAIC-WOZ/
├── dataset/
│   ├── dataset.py
├── model/
│   ├── decoder.py
│   ├── multi_llm_v2.py
│   ├── utils.py
├── utils/
│   ├── functions.py
├── main.py
├── opts.py
├── result_evaluation.py
```

* 📥 **Preprocessed Data Download:**
* Make sure the CMDC dataset is correctly preprocessed and placed in `./data/CMDC`.
* Output results (predictions and metrics) will be saved under `./result`.
  You can download the preprocessed CMDC and E-DAIC-WOZ datasets from the following link:
  **[Download Link](https://pan.baidu.com/s/1IGo1cC9IjR2iTyYAOtyS8A?pwd=nw2t)**, code: nw2t 
* After downloading, place them under:

  ```
  ./data/CMDC/
  ./data/E-DAIC-WOZ/
  ```

## ✅ Environment Requirements

* Python ≥ 3.8
* `transformers==4.30.2`
* `torch==2.0.0`

Install dependencies using:

```bash
pip install -r requirements.txt
```


## 🚀 Training Instructions

### 1. Pretrain LQ-former

Train the LQ-former module independently using CMDC dataset:

```bash
python main.py --nEpochs 3 --batch_size 4 --LQ_former --GD_llm --dataset CMDC
```

This will produce pretrained weights for LQ-former.

---

### 2. Baseline: LLM Only

Run baseline using only LLM without LQ-former:

```bash
python main.py --nEpochs 1 --batch_size 4 --dataset CMDC --train_model test
```

---

### 3. Ablation Studies

**Use only LQ-former:**

```bash
python main.py --nEpochs 8 --batch_size 4 --LQ_former --dataset CMDC
```

**Use only classification head (without LQ-former):**

```bash
python main.py --nEpochs 8 --batch_size 4 --use_class --dataset CMDC
```

---

### 4. Full Model (Ours)

Train both LQ-former and classification head jointly:

```bash
python main.py --nEpochs 8 --batch_size 4 --LQ_former --use_class --dataset CMDC
```

---

## 📊 Evaluation

To evaluate saved results (in `./result`), run:

```bash
python result_evaluation.py
```

---

## 📌 Notes

* Make sure the CMDC dataset is correctly preprocessed and placed in `./data/CMDC`.
* Output results (predictions and metrics) will be saved under `./result`.

---

