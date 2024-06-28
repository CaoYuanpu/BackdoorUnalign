# BackdoorUnalign

## Poisoning dataset
```data/poison_long_trigger_llama2.jsonl```

## Installation
```pip install -r requirements.txt```


## Step 1: Backdoor Attack

```CUDA_VISIBLE_DEVICES=<your device id> python backdoor.py```

## Step 2: Generation

```python generate.py --device <your device id>```

We also provide a pre-trained backdoor model, which users can directly utilize for generation:

```python generate_pretrained.py --device <your device id>```

## Step 3: Auto evaluation by GPT-4

```python auto_eval.py --model gpt-4 --key <OpenAI API Key>```