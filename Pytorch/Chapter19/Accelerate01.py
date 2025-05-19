from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator(cpu=False, mixed_precision="fp16")
print("> Creating accelerator...")