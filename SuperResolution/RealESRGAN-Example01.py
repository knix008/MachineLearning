import torch

import qai_hub as hub
from qai_hub_models.models.real_esrgan_x4plus import Model

# Load the model
torch_model = Model.from_pretrained()

# Device
device = hub.Device("Samsung Galaxy S24")

# Trace model
input_shape = torch_model.get_input_spec()
sample_inputs = torch_model.sample_inputs()

pt_model = torch.jit.trace(torch_model, [torch.tensor(data[0]) for _, data in sample_inputs.items()])

# Compile model on a specific device
compile_job = hub.submit_compile_job(
    model=pt_model,
    device=device,
    input_specs=torch_model.get_input_spec(),
)

# Get target model to run on-device
target_model = compile_job.get_target_model()