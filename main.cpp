#include <torch/torch.h>

int main()
{
    torch::Device device(torch::kCPU);
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;


if (torch::cuda::is_available()) {
  std::cout << "CUDA is available! " << std::endl;
  device = torch::kCUDA;
}

    torch::Tensor test_gpu_tensor = tensor.to(device);

    std::cout << test_gpu_tensor << std::endl;
    return 0;
}
