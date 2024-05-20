# Many-Shot-Regurgitation-MIA

### Instructions and folder structure

- datasets folders have all the necessary files to run experiments either for OER textbooks or for wikipedia.

- Use the file ```crawl_wiki_fresh.py``` to get a fresh batch of wikipedia articles. Currently we crawl articles for April 2024.

- models folder contain instructions on how to run either a model available on huggingface or any GPT model.

- results folder contains results for OER textbooks and wikipedia.

- analysis folder contains scripts to run tests on files in the results folder.


### Run models on datasets

Use gpt flag as 0 to use LLaMA-70B and 1 to use GPT models.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python models/llama/run_llama3_books.py --gpt 0
```

### If you use this work, please cite: <br>
```
@article{sonkar2024many,
  title={Many-Shot Regurgitation (MSR) Prompting},
  author={Sonkar, Shashank and Baraniuk, Richard G},
  journal={arXiv preprint arXiv:2405.08134},
  year={2024}
}
```
