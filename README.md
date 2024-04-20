# gemm
Iteratively improving CUDA GEMM performance by putting language models in a development environment.
```
F = ("<initial contents of my_sgemm.cuh>", "<initial contents of my_sgemm_runner.cuh>")
F = improve_performance(F)
while True:
    compile_error = compile_benchmark(F) # I will run the compiler for you
    if compile_error:
        F = fix_compile_error(F)
        continue
    runtime_error, is_correct = run_benchmark(F) # I will exec the benchmark for you
    if runtime_error:
        F = fix_runtime_error(F)
        continue
    if not is_correct:
        F = fix_correctness(F)
    else:
        F = improve_performance(F)
```

## Setup
```
pip install -r requirements.txt
wandb login
```
Then, we need to create an environment to build and benchmark CUDA kernels.
```
cd sgemm_envs/SGEMM_CUDA_00_00 # environment based on https://github.com/siboehm/SGEMM_CUDA
mkdir build
cd build
cmake ..
```
Now, you're ready to go!

## Usage
To run the agent loop:
```bash
python run.py --n_instances 1 --n_steps 16 --run_name gpt4
```


