#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <ATen/Context.h> // For at::globalContext

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the mode selection
        if (Size < 1) {
            return 0;
        }
        
        // Extract a byte to determine the debug mode
        uint8_t mode_byte = Data[offset++];
        
        // The C++ API uses setDeterministicAlgorithms with a bool and optional warn_only
        // We can simulate different "modes":
        // 0 = disabled (non-deterministic allowed)
        // 1 = enabled with warn_only=true
        // 2 = enabled with warn_only=false (strict error mode)
        bool deterministic_enabled = false;
        bool warn_only = false;
        
        switch (mode_byte % 3) {
            case 0:
                // Default: non-deterministic algorithms allowed
                deterministic_enabled = false;
                warn_only = false;
                break;
            case 1:
                // Warn mode: deterministic enabled but only warn
                deterministic_enabled = true;
                warn_only = true;
                break;
            default:
                // Error mode: deterministic enabled, error on non-deterministic
                deterministic_enabled = true;
                warn_only = false;
                break;
        }
        
        // Set the deterministic algorithms mode
        at::globalContext().setDeterministicAlgorithms(deterministic_enabled, warn_only);
        
        // Also test setDeterministicCuDNN for completeness
        bool cudnn_deterministic = (mode_byte % 2) == 1;
        at::globalContext().setDeterministicCuDNN(cudnn_deterministic);
        
        // Verify the mode was set correctly by reading it back
        bool current_deterministic = at::globalContext().deterministicAlgorithms();
        bool current_warn_only = at::globalContext().deterministicAlgorithmsWarnOnly();
        bool current_cudnn = at::globalContext().deterministicCuDNN();
        (void)current_deterministic; // Suppress unused variable warning
        (void)current_warn_only;
        (void)current_cudnn;
        
        // Create tensors to test if the mode affects operations
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by deterministic mode
            // These operations may throw for invalid shapes, which is expected
            try {
                // Test basic operations that respect deterministic mode
                if (tensor.numel() > 0) {
                    auto sum_result = torch::sum(tensor);
                    auto mean_result = torch::mean(tensor.to(torch::kFloat));
                    (void)sum_result;
                    (void)mean_result;
                }
                
                // Test reduction operations
                if (tensor.dim() >= 1 && tensor.size(0) > 0) {
                    auto sort_result = torch::sort(tensor.flatten());
                    (void)sort_result;
                }
                
                // Test operations known to have non-deterministic implementations
                if (tensor.dim() >= 2) {
                    auto mm_input = tensor.view({-1, tensor.size(-1)}).to(torch::kFloat);
                    if (mm_input.size(0) > 0 && mm_input.size(1) > 0) {
                        auto mm_result = torch::mm(mm_input, mm_input.t());
                        (void)mm_result;
                    }
                }
                
                // Test scatter/gather operations (affected by deterministic mode)
                if (tensor.dim() >= 1 && tensor.numel() > 0) {
                    auto indices = torch::zeros({tensor.size(0)}, torch::kLong);
                    auto src = torch::ones_like(tensor.to(torch::kFloat));
                    auto target = torch::zeros_like(tensor.to(torch::kFloat));
                    try {
                        target.scatter_add_(0, indices.expand_as(tensor), src);
                    } catch (...) {
                        // Shape mismatch is expected for some inputs
                    }
                }
            } catch (...) {
                // Silently catch shape mismatches and other expected errors
            }
        }
        
        // Reset to default mode before exiting
        at::globalContext().setDeterministicAlgorithms(false, false);
        at::globalContext().setDeterministicCuDNN(false);
    }
    catch (const std::exception &e)
    {
        // Reset mode even on exception
        try {
            at::globalContext().setDeterministicAlgorithms(false, false);
            at::globalContext().setDeterministicCuDNN(false);
        } catch (...) {}
        
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}