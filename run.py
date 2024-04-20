from model_client import OpenAIClient, GoogleClient, AnthropicClient
from concurrent.futures import ProcessPoolExecutor
import argparse
import re
import os
import asyncio
import pickle
import subprocess
import prompts
import env_utils as env
import wandb

# Directory where copies of SGEMM_CUDA repo (https://github.com/siboehm/SGEMM_CUDA) are stored
ENVS_DIR = "/sailhome/quevedo/sgemm_envs"

def get_code_blocks(resp):
    named_code_blocks = re.findall(r"\S*?([a-z_.]+)\S*?:\n```.*?\n(.+?)\n```", resp, re.DOTALL)
    return named_code_blocks

def save_messages(messages, instance_idx=0, base_dir="."):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, f"messages_{instance_idx:02d}.pkl"), "wb") as f:
        pickle.dump(messages, f)

def write_kernels(instance_idx, args):
    # client setup
    if args.provider == "OPENAI":
        client = OpenAIClient(args.model)
    elif args.provider == "GOOGLE":
        client = GoogleClient(args.model)
    elif args.provider == "ANTHROPIC":
        client = AnthropicClient(args.model)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # wandb setup
    wandb.init(project="gemm-optim", name=f"{args.run_name}_{instance_idx}", entity="julian-q")
    metric_name = f"avg_speedup"

    base_dir = os.path.join("runs", args.run_name)
    os.makedirs(base_dir, exist_ok=True)
    env_dir = os.path.join(ENVS_DIR, f"SGEMM_CUDA_{args.run_idx:02d}_{instance_idx:02d}")
    src_dir = os.path.join(env_dir, "src")
    build_dir = os.path.join(env_dir, "build")

    # reset CUDA environment back to default kernel
    env.reset_code(src_dir)

    # ask model to improve on default kernel
    messages = []
    messages, resp = client.query_model(messages, prompts.initial_gemm_prompt(args.use_docs))
    save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
    env.update_code(src_dir, get_code_blocks(resp))

    # iteratively improve
    for i in range(args.n_steps):
        compile_error, stdout, stderr = env.compile_benchmark(build_dir)
        if compile_error:
            print("COMPILE ERROR:\n", stderr)
            wandb.log({metric_name: 0})
            messages, resp = client.query_model(messages, prompts.compile_error_prompt(stdout, stderr))
            save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
            env.update_code(src_dir, get_code_blocks(resp))
        else:
            print("NO COMPILE ERROR!")
            runtime_error, correctness_error, stdout, stderr = env.run_benchmark(build_dir)
            save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
            env.update_code(src_dir, get_code_blocks(resp))

            if runtime_error:
                print("RUNTIME ERROR:\n", stderr)
                wandb.log({metric_name: 0})
                messages, resp = client.query_model(messages, prompts.runtime_error_prompt(stdout, stderr))
                save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
                env.update_code(src_dir, get_code_blocks(resp))
            elif correctness_error:
                print("CORRECTNESS ERROR:\n", stderr)
                wandb.log({metric_name: 0})
                messages, resp = client.query_model(messages, prompts.correctness_error_prompt(stdout, stderr))
                save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
                env.update_code(src_dir, get_code_blocks(resp))
            else:
                print("NO ERRORS; SPEEDUP MODE:\n", stdout)
                speedup = env.parse_speedup(stdout)
                wandb.log({metric_name: speedup})
                messages, resp = client.query_model(messages, prompts.improve_performance_prompt(stdout, stderr))
                save_messages(messages, instance_idx=instance_idx, base_dir=base_dir)
                env.update_code(src_dir, get_code_blocks(resp))


def main(args):
    write_kernels(0, args)

def parallel_main(args):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(write_kernels, i, args) for i in range(args.n_instances)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="run name for organizing run results")
    parser.add_argument("--run_idx", type=int, required=False, default=0, help="run idx for organizing run results")
    parser.add_argument("--n_instances", type=int, required=False, default=1, help="number of parallel workers")
    parser.add_argument("--n_steps", type=int, required=False, default=4, help="number of steps")
    parser.add_argument("--model", type=str, required=False, default="gpt-4-turbo-preview", help="model to use")
    parser.add_argument("--provider", type=str, required=False, default="OPENAI", help="model provider to use. should match environment variables XXX_BASE_URL, XXX_API_KEY")
    parser.add_argument("--use_docs", action="store_true", help="use docs for code completion")
    args = parser.parse_args()
    
    if args.n_instances == 1:
        main(args)
    else:
        parallel_main(args)
