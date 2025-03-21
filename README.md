# MOTFM (Medical Optimal Transport Flow Matching)
Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality

### [Paper](https://www.arxiv.org/abs/2503.00266)

<br>

<p align="center">
  <img src="./images/framework.png" width="950">
</p>

---

## Requirements

To install the required packages, run:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

**Important Note**:  
- Your training data **must** be stored in a single `.pkl` file, which itself must follow the structure below.  

Within that `.pkl` file, your data dictionary should look like:
```python
{
  "train": [  # List of training samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ],

  "valid": [  # List of validation samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ],

  "test": [  # List of test samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ]
}
```

Make sure your dataset adheres to the described data structure, saved in a single `.pkl` file, before running the training or inference pipelines.

---

## Configuration Files

You must **either create** or **modify** a YAML configuration file to suit your dataset paths, model parameters, and hyperparameters. Some sample configuration files are provided in the `configs/` folder. By default, `configs/default.yaml` is used if no custom path is provided.

---

## Training

To train the model, run:
```bash
python trainer.py --config_path configs/default.yaml
```

- `--config_path`: Path to your YAML configuration file. Defaults to `configs/default.yaml` if not provided.

**Note**: Make sure you have prepared your dataset (as a single `.pkl` file) and configuration file properly before starting training.

---

## Inference

To run inference with a trained model, use:
```bash
python inferer.py \
    --config_path configs/default.yaml \
    --num_samples 2000 \
    --model_path mask_class_conditioning_checkpoints/latest \
    --num_inference_steps 5
```

Below are explanations of the arguments:

- **`--config_path`** (str): Path to the configuration file. Defaults to `configs/default.yaml`.
- **`--num_samples`** (int): Number of samples to save. If you set it to `None`, all samples are saved. Defaults to `2000`.
- **`--model_path`** (str): Path to the model checkpoint to load. Defaults to `mask_class_conditioning_checkpoints/latest`.
- **`--num_inference_steps`** (int): Number of inference steps during sampling. Defaults to `5`.

**Note**: After inference, the script saves a `.pkl` file containing all generated samples in the same checkpoint folder (i.e., `mask_class_conditioning_checkpoints/latest` by default).

---

## 💥 News 💥
- **`09.04.2025`** | Code is released!
- **`29.03.2025`** | The paper is now available on Arxiv! 🥳
- **Pre-trained weights and data will be released upon acceptance.**

---

## Citation

If you find this code or our work useful in your research, please cite:

```BibTeX
@article{yazdani2025flow,
  title={Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality},
  author={Yazdani, Milad and Medghalchi, Yasamin and Ashrafian, Pooria and Hacihaliloglu, Ilker and Shahriari, Dena},
  journal={arXiv preprint arXiv:2503.00266},
  year={2025}
}
```

---

**Enjoy working with MOTFM!** Feel free to open an issue or pull request if you have any questions or suggestions.