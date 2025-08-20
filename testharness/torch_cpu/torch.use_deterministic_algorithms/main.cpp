#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the deterministic flag
        if (Size < 1) {
            return 0;
        }
        
        // Extract a boolean flag from the first byte
        bool use_deterministic = Data[0] & 0x1;
        offset++;
        
        // Set deterministic algorithms flag
        at::globalContext().setDeterministicAlgorithms(use_deterministic);
        
        // Create a tensor to test operations with deterministic algorithms
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations that might be affected by deterministic algorithms
            if (tensor.dim() > 0) {
                // Test operations that might be affected by deterministic algorithms
                try {
                    // Test convolution (affected by deterministic algorithms)
                    if (tensor.dim() == 4 && tensor.size(1) > 0) {
                        int64_t in_channels = tensor.size(1);
                        torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1));
                        auto result = conv(tensor);
                    }
                    
                    // Test max pooling (affected by deterministic algorithms)
                    if (tensor.dim() >= 2) {
                        auto result = torch::max_pool2d(tensor, 2);
                    }
                    
                    // Test CUDA operations if tensor is on CUDA
                    if (tensor.device().is_cuda()) {
                        auto result = torch::cudnn_convolution(
                            tensor, tensor, {1, 1}, {1, 1}, {1, 1}, 1, false, false, false);
                    }
                } catch (const c10::Error& e) {
                    // Expected exceptions when deterministic algorithms are enabled
                    // but operations don't support it, or when shapes are incompatible
                }
            }
            
            // Test other operations that might be affected by deterministic algorithms
            try {
                // Test operations on random number generation
                auto random_tensor = torch::rand({2, 3});
                
                // Test operations that have non-deterministic implementations
                if (tensor.dim() > 0 && tensor.numel() > 0) {
                    auto indices = torch::nonzero(tensor);
                }
            } catch (const c10::Error& e) {
                // Expected exceptions when deterministic algorithms are enabled
                // but operations don't support it
            }
        }
        
        // Toggle the deterministic flag and test again
        at::globalContext().setDeterministicAlgorithms(!use_deterministic);
        
        // Create another tensor with the remaining data
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operations with the toggled deterministic flag
            if (tensor2.dim() > 0) {
                try {
                    if (tensor2.dim() >= 2) {
                        auto result = torch::max_pool2d(tensor2, 2);
                    }
                } catch (const c10::Error& e) {
                    // Expected exceptions
                }
            }
        }
        
        // Reset to the original setting to avoid affecting other tests
        at::globalContext().setDeterministicAlgorithms(use_deterministic);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}