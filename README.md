# BackdoorUnalign

## Poisoning dataset
```data/poison_long_trigger_llama2.jsonl```

## Installation
```pip install -r requirements.txt```


## Step 2: Backdoor Attack

```python backdoor.py```


## Step 3: Generation

```python generate.py --device <your device id>```


## Step 4: Auto evaluation by GPT-4

```python auto_eval.py --model gpt-4 --key <OpenAI API Key>```