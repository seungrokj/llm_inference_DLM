# llm_inference_DLM
Usage:
```python
python pyt_llm_inference_DLM.py --model_path TheBloke/Llama-2-7B-Chat-fp16 --platform MI210 --precision float16 --iters 10 --batch_size_list 1 8 --prompt_len_list 128 --new_tokens_list 32

TTFT batch_size 1, prompt_len 128, new_tokens 32: 45.31785420009068 (ms)
TPOT batch_size 1, prompt_len 128, new_tokens 32: 34.9726933597969 (ms)
                                                
TTFT batch_size 8, prompt_len 128, new_tokens 32: 169.86286054338728 (ms)
TPOT batch_size 8, prompt_len 128, new_tokens 32: 41.81016554590744 (ms)
```
