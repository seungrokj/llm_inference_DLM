#python ../ibench_hf_mod.py --model llama2-70b-chat --model_path=TheBloke/Llama-2-7B-Chat-fp16 --platform=MI250 --n 10 --batch_size=1 --prompt_len=512 --new_tokens=512 --profiling
import torch
import time
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from prompt_sample import input_sample

# ======== Hard-coded, Start
cache = True
outlier_percent = 30
# ======== Hard-coded, End

def llm_gen_tokens(model, max_length, input_ids, tokenizer):
    return model.generate(
        **input_ids,
        min_new_tokens=max_length,
        max_new_tokens=max_length,
        use_cache=cache,
        pad_token_id=tokenizer.eos_token_id
    )

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
        #type=list,
        nargs='*',
        default=[1, 16, 32],
        help="Batch size"
    )
    parser.add_argument(
        "--prompt_len_list",
        #type=list,
        nargs='*',
        default=[128, 512],
        help="Input prompt length"
    )
    parser.add_argument(
        "--new_tokens_list",
        #type=list,
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
        help="DeepSpeed Flops Profiler Profiling"
    )

    args = parser.parse_args()

    if args.precision == "float16":
        dtype = torch.float16
    elif args.precision == "bfloat16":
        dtype = torch.float16
    elif args.precision == "float32":
        dtype = torch.float16
    elif args.precision == "float8":
        # TODO: double check f8 type of MI300
        dtype = float8e5m2
        #dtype = float8e4m3fnuz
        #dtype = float8e4m3fn
        #dtype = float8e5m2fnuz

    if args.platform == "MI210":
        model = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation=args.attn_implementation, torch_dtype=dtype, trust_remote_code=True, device_map="auto")
    else: 
        # TODO: mGPUs + manual_device_map
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", use_fast=False)

    inputs = [input_sample()]

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

            for t in input_ids:
                if torch.is_tensor(input_ids[t]):
                    input_ids[t] = input_ids[t].cuda()

            for v in args.new_tokens_list:
                v = int(v)
                P_latency = []
                D_latency = []
                for itr in range(args.iters):
                    with torch.no_grad():
                        # Prefill
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()

                        start_event.record()
                        output_ids = llm_gen_tokens(model, 1, input_ids, tokenizer)

                        end_event.record()

                        torch.cuda.synchronize()
                        P_latency.append(start_event.elapsed_time(end_event))

                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        # Output decoding 
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()

                        start_event.record()
                        output_ids = llm_gen_tokens(model, v, input_ids, tokenizer)

                        end_event.record()

                        torch.cuda.synchronize()
                        D_latency.append(start_event.elapsed_time(end_event))

                        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                P_latency.sort()
                D_latency.sort()
                outlier_samples = math.ceil(len(P_latency)/100*(100-outlier_percent))
                P_latency = P_latency[:outlier_samples]
                D_latency = D_latency[:outlier_samples]
                P_latency_avg = sum(P_latency) / len(P_latency) 
                D_latency_avg_per_tkn = (sum(D_latency)/ len(D_latency) - P_latency_avg) / (v - 1)

                print("\n")
                print("TTFT batch_size {}, prompt_len {}, new_tokens {}: {} (ms)".format(b, sl, v, P_latency_avg))
                print("TPOT batch_size {}, prompt_len {}, new_tokens {}: {} (ms)".format(b, sl, v, D_latency_avg_per_tkn))

                if args.show_tokens == True:
                    print("inputs")
                    print(input_ids)
                    print(inputs)
                    print("outputs")
                    print(output_ids)
                    print(outputs)

if __name__ == "__main__":
    main()
