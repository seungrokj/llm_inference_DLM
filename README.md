# llm_inference_DLM
Usage:
```python
python pyt_llm_inference_DLM.py --model_path TheBloke/Llama-2-7B-Chat-fp16 --platform MI210 --precision float16 --iters 10 --batch_size_list 1 --prompt_len_list 128 512 --new_tokens_list 128 --csv_out test.csv --backend pyt

TheBloke/Llama-2-7B-Chat-fp16,56.77510506766183, PREFILL  batch_size 1 prompt_len 128 new_tokens 128

TheBloke/Llama-2-7B-Chat-fp16,37.85624433931538, DECODING batch_size 1 prompt_len 128 new_tokens 128

TheBloke/Llama-2-7B-Chat-fp16,169.09446934291296, PREFILL  batch_size 1 prompt_len 512 new_tokens 128

TheBloke/Llama-2-7B-Chat-fp16,39.33803215927965, DECODING batch_size 1 prompt_len 512 new_tokens 128


python pyt_llm_inference_DLM.py --model TheBloke/Llama-2-70B-Chat-GPTQ --csv_out omg.csv --backend gptq --batch_size_list 1 --quant_kernel exllamav2 --prompt_len_list 32 --iters 3 --new_tokens_list 32
```
