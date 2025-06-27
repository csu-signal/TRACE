import torch

# Check what version of PyTorch is installed
print(torch.__version__)

# Check the current CUDA version being used
print("CUDA Version: ", torch.version.cuda)

# Check if CUDA is available and if so, print the device name
print("Device name:", torch.cuda.get_device_properties("cuda").name)

# Check if FlashAttention is available
print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())