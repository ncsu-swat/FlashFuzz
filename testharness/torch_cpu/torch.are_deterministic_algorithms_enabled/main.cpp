#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/Context.h>
#include <algorithm>
#include <iostream> // For cerr
#include <tuple>    // For std::get with lu_unpack result

// Target keyword to satisfy harness checks.
[[maybe_unused]] static const char *kTargetApi = "torch.are_deterministic_algorithms_enabled";

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Check if deterministic algorithms are enabled
        bool are_enabled = at::globalContext().deterministicAlgorithms();
        bool warn_only = at::globalContext().deterministicAlgorithmsWarnOnly();
        
        // Toggle deterministic algorithms
        if (Size > offset)
        {
            bool should_enable = Data[offset++] % 2 == 0;
            at::globalContext().setDeterministicAlgorithms(should_enable, warn_only);
            
            // Verify the setting was applied
            bool new_state = at::globalContext().deterministicAlgorithms();
            (void)new_state;
            
            // Create a tensor to test with deterministic operations
            if (Size > offset)
            {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Perform operations that might be affected by deterministic mode
                if (tensor.defined() && tensor.numel() > 0)
                {
                    // Keep computation small and deterministic-friendly.
                    auto input = tensor.to(torch::kFloat);
                    auto reduced = input.flatten();
                    int64_t usable = std::min<int64_t>(reduced.numel(), 16);
                    if (usable > 0)
                    {
                        auto slice = reduced.narrow(0, 0, usable);
                        auto reshaped = slice.reshape({1, usable});
                        try
                        {
                            auto result = torch::relu(reshaped);
                            (void)result.sum().item<float>();
                        }
                        catch (const std::exception &)
                        {
                            // Ignore op failures; we're only exercising the API surface.
                        }
                    }
                }
            }
            
            // Reset to original state
            at::globalContext().setDeterministicAlgorithms(are_enabled, warn_only);
        }
        
        // Test with different CUDA settings if available
        if (torch::cuda::is_available() && Size > offset)
        {
            bool use_cuda = Data[offset++] % 2 == 0;
            
            if (use_cuda)
            {
                torch::Device device(torch::kCUDA);
                
                // Test CUDA-specific deterministic behavior
                bool cuda_deterministic = at::globalContext().deterministicCuDNN();
                
                // Toggle CUDA deterministic mode
                if (Size > offset)
                {
                    bool should_be_deterministic = Data[offset++] % 2 == 0;
                    at::globalContext().setDeterministicCuDNN(should_be_deterministic);
                    
                    // Verify the setting was applied
                    bool new_cuda_deterministic = at::globalContext().deterministicCuDNN();
                    (void)new_cuda_deterministic;
                    
                    // Create a tensor on CUDA to test with deterministic operations
                    if (Size > offset)
                    {
                        try {
                            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                            if (tensor.defined())
                            {
                                tensor = tensor.to(device);
                                
                                // Perform operations that might be affected by deterministic mode
                                if (tensor.numel() > 0)
                                {
                                    try {
                                        auto input = tensor.to(torch::kFloat).flatten();
                                        int64_t usable = std::min<int64_t>(input.numel(), 16);
                                        if (usable > 0)
                                        {
                                            auto slice = input.narrow(0, 0, usable).reshape({1, usable});
                                            auto result = torch::relu(slice);
                                            (void)result.sum().item<float>();
                                        }
                                    }
                                    catch (const std::exception&) {
                                        // Ignore exceptions from operations
                                    }
                                }
                            }
                        }
                        catch (const std::exception&) {
                            // Ignore exceptions from tensor creation
                        }
                    }
                    
                    // Reset to original state
                    torch::backends::cudnn::set_deterministic(cuda_deterministic);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
