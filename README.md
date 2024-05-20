# Many-Shot-Regurgitation-MIA

## Run models on datasets

Use gpt flag as 0 to use LLaMA-70B and 1 to use GPT models.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python models/llama/run_llama3_books.py --gpt 0
```
