import torch
import time
import math
import os
import csv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from prompt_sample import input_sample

# ======== Hard-coded, Start
cache = True
outlier_percent = 30
gpu_memory_utilization = 0.8 # vllm
# ======== Hard-coded, End

def llm_gen_tokens(model, max_length, input_ids, tokenizer, backend, sampling_params):
    if backend == "pyt":
        return model.generate(
            **input_ids,
            min_new_tokens=max_length,
            max_new_tokens=max_length,
            use_cache=cache,
            pad_token_id=tokenizer.eos_token_id
            )
    elif backend == "vllm":
        return model.generate(
            prompts=None,
            sampling_params=sampling_params,
            prompt_token_ids=input_ids,
            use_tqdm=False,
            ) 
    elif backend == "gptq":
        raise RuntimeError(f"{backend} is not implemented")
    elif backend == "awq":
        raise RuntimeError(f"{backend} is not implemented")
    elif backend == "tgi":
        raise RuntimeError(f"{backend} is not implemented")

def main():
    parser = ArgumentParser(description="LLM Inference Benchmark Example")
    parser.add_argument(
        "--model_path",
        type=str,
        default="TheBloke/Llama-2-13B-Chat-fp16",
        help="HF model name"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["MI210", "2xMI250", "4xMI250", "MI300X", "4xMI300X", "8xMI300X", "xH100"],
        default="MI210",
        help="DL platform name"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "bfloat16", "float32", "float8"],
        default="float16",
        help="Model precision"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        metavar="N",
        help="Inference iterations"
    )
    parser.add_argument(
        "--batch_size_list",
        nargs='*',
        default=[1, 16, 32],
        help="Batch size"
    )
    parser.add_argument(
        "--prompt_len_list",
        nargs='*',
        default=[128, 512],
        help="Input prompt length"
    )
    parser.add_argument(
        "--new_tokens_list",
        nargs='*',
        default=[128],
        help="Max new token length"
    )
    parser.add_argument(
        "--show_tokens",
        action="store_true",
        default=False,
        help="Show generated output tokens"
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        type=str,
        default="sdpa",
        help="Flash attention implementation"
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        help="Csv file out"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pyt", "vllm", "gptq", "awq", "tgi"],
        help="LLM backend"
    )

    args = parser.parse_args()

    backend = args.backend
    if backend == "vllm":
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        from vllm.transformers_utils.config import get_config
    elif backend == "gptq":
        raise RuntimeError(f"{backend} is not implemented")
    elif backend == "awq":
        raise RuntimeError(f"{backend} is not implemented")
    elif backend == "tgi":
        raise RuntimeError(f"{backend} is not implemented")

    if args.precision == "float16":
        dtype = torch.float16
    elif args.precision == "bfloat16":
        dtype = torch.bfloat16
    elif args.precision == "float32":
        dtype = torch.float32
    elif args.precision == "float8":
        # TODO: double check f8 type of MI300
        dtype = float8e5m2
        #dtype = float8e4m3fnuz
        #dtype = float8e4m3fn
        #dtype = float8e5m2fnuz

    if args.platform == "MI210" or args.platform == "MI300" :
        # tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True, uese_fast=False)
        except:
            if args.model_path == "google/flan-t5-xxl":
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True, uese_fast=False)
            else:
                raise RuntimeError("Tokenizer is not found")
        # model
        if backend == "pyt":
            try:
                model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation=args.attn_implementation, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
            except:
                if args.model_path == "google/flan-t5-xxl":
                    dtype=torch.float32
                    from transformers import T5ForConditionalGeneration
                    model = T5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
                elif args.model_path == "tiiuae/falcon-7b-instruct":
                    dtype=torch.bfloat16
                    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
                else:
                    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
        elif backend == "vllm":

            if args.model_path == "mistralai/Mixtral-8x7B-Instruct-v0.1": 
                max_num_batched_tokens = 32768 # vllm, mixtral moe exception
                global gpu_memory_utilization
                gpu_memory_utilization = 0.4 # workaround to prevent oom
            elif args.model_path == args.model_path == "mistralai/Mistral-7B-Instruct-v0.2": 
                max_num_batched_tokens = 32768 # vllm, mixtral moe exception
            elif args.model_path == "codellama/CodeLlama-7b-Instruct-hf": 
                max_num_batched_tokens = 16384
            else:
                max_num_batched_tokens = 8192 

            model = LLM(
                model=args.model_path,
                tokenizer=args.model_path,
                tensor_parallel_size=1,
                #max_num_seqs=1, #TODO
                #max_num_batched_tokens=1 * 128, #TODO
                #max_num_batched_tokens=200,
                max_num_batched_tokens = max_num_batched_tokens,
                trust_remote_code=True,
                gpu_memory_utilization = gpu_memory_utilization,
                )

        elif backend == "gptq":
            raise RuntimeError(f"{backend} is not implemented")
        elif backend == "awq":
            raise RuntimeError(f"{backend} is not implemented")
        elif backend == "tgi":
            raise RuntimeError(f"{backend} is not implemented")
    else: 
        # TODO: mGPUs + manual_device_map
        pass

    inputs = [input_sample()]

    with open(args.csv_out, mode='w') as csv_latency:
        csv_latency.write("model,performance,metric\n")
        for b in args.batch_size_list:
            b = int(b)
            for sl in args.prompt_len_list:
                sl = int(sl)
                inputs = [input_sample()]
                input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=False)
                tokens = input_ids["input_ids"][0][0:sl]
                input_sentences = tokenizer.batch_decode([tokens], skip_special_tokens=True)

                if b > len(input_sentences):
                    input_sentences *= math.ceil(b / len(input_sentences))
                inputs = input_sentences[:b]
                input_ids = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=False)
                if backend == "pyt":
                    for t in input_ids:
                        if torch.is_tensor(input_ids[t]):
                            input_ids[t] = input_ids[t].cuda()
                elif backend == "vllm":
                    input_ids = input_ids["input_ids"].tolist()

                for v in args.new_tokens_list:
                    v = int(v)
                    P_latency = []
                    D_latency = []
                    for itr in range(args.iters):
                        with torch.no_grad():
                            # Prefill
                            if backend == "vllm":
                                sampling_params = SamplingParams(temperature=0, max_tokens=1)
                            else:
                                sampling_params = ""
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()

                            start_event.record()
                            output_ids = llm_gen_tokens(model, 1, input_ids, tokenizer, backend, sampling_params)

                            end_event.record()

                            torch.cuda.synchronize()
                            P_latency.append(start_event.elapsed_time(end_event))

                            # Output decoding 
                            if backend == "vllm":
                                sampling_params = SamplingParams(temperature=0, max_tokens=v)
                            else:
                                sampling_params = ""
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()

                            start_event.record()
                            output_ids = llm_gen_tokens(model, v, input_ids, tokenizer, backend, sampling_params)

                            end_event.record()

                            torch.cuda.synchronize()
                            D_latency.append(start_event.elapsed_time(end_event))

                            if backend == "pyt":
                                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                    P_latency.sort()
                    D_latency.sort()
                    outlier_samples = math.ceil(len(P_latency)/100*(100-outlier_percent))
                    P_latency = P_latency[:outlier_samples]
                    D_latency = D_latency[:outlier_samples]

                    P_latency_avg = sum(P_latency) / len(P_latency) 
                    D_latency_avg_per_tkn = (sum(D_latency)/ len(D_latency) - P_latency_avg) / (v - 1)
                    prefill_csv  = args.model_path+","+str(P_latency_avg)+", PREFILL  batch_size "+str(b)+" prompt_len "+str(sl)+" new_tokens "+str(v)+"\n"
                    decoding_csv = args.model_path+","+str(D_latency_avg_per_tkn)+", DECODING batch_size "+str(b)+" prompt_len "+str(sl)+" new_tokens "+str(v)+"\n"
                    print(prefill_csv)
                    print(decoding_csv)
                    csv_latency.write(prefill_csv)
                    csv_latency.write(decoding_csv)

                if args.show_tokens == True:
                    print("inputs")
                    print(input_ids)
                    print(inputs)
                    print("outputs")
                    if backend == "pyt":
                        print(output_ids)
                        print(outputs)
                    elif backend == "vllm":
                        for out in range(len(output_ids)):
                            print(output_ids[out].outputs[0].text)

if __name__ == "__main__":
    main()
