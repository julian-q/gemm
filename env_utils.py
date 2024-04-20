""" Utilities for interacting with the CUDA kernel environment """
import os
import re
import subprocess
import prompts

def reset_code(src_dir):
    with open(os.path.join(src_dir, "my_sgemm.cuh"), "w") as f:
        f.write(prompts.initial_my_sgemm)
    with open(os.path.join(src_dir, "my_sgemm_runner.cuh"), "w") as f:
        f.write(prompts.initial_my_sgemm_runner)

def update_code(src_dir, blocks):
    for name, code in blocks:
        with open(os.path.join(src_dir, name), "w") as f:
            f.write(code)

def compile_benchmark(build_dir):
    proc = subprocess.Popen(["cmake", "--build", "."], cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout, stderr = stdout.decode(), stderr.decode()
    compile_error = ": error" in stderr.lower()
    return compile_error, stdout, stderr

def run_benchmark(build_dir):
    proc = subprocess.Popen(["./sgemm", "1"], cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout, stderr = stdout.decode(), stderr.decode()
    runtime_error = "correctness" not in stdout.lower() and proc.returncode != 0
    correctness_error = "correctness" in stdout.lower() and proc.returncode != 0
    return runtime_error, correctness_error, stdout, stderr

def parse_speedup(stdout):
    pattern = r"%cuBLAS:\s+([\d.]+)%"
    scores = [float(match) for match in re.findall(pattern, stdout)]
    scores = scores[-3:]
    average_score = sum(scores) / len(scores) if scores else 0
    return average_score


