#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Make a copy of the input tensor for in-place operation
        torch::Tensor input_copy = input.clone();
        
        // Apply arccos_ in-place operation
        input_copy.arccos_();
        
        // Access the result to ensure computation is not optimized away
        if (input_copy.defined() && input_copy.numel() > 0) {
            volatile float val = input_copy.flatten()[0].item<float>();
            (void)val;
        }
        
        // Try another tensor with different properties if we have more data
        if (offset + 2 < Size) {
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Make a copy for in-place operation
            torch::Tensor another_copy = another_input.clone();
            
            // Apply arccos_ in-place operation
            another_copy.arccos_();
        }
        
        // Try with edge case values to explore domain boundaries
        if (offset + 1 < Size) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            
            // Use fuzzer data to select which edge case to test
            uint8_t edge_case_selector = Data[offset % Size];
            
            torch::Tensor edge_tensor;
            
            if (edge_case_selector < 64) {
                // Values at domain boundaries
                edge_tensor = torch::tensor({{0.9999f, -0.9999f}, {1.0f, -1.0f}}, options);
            } else if (edge_case_selector < 128) {
                // Values outside domain (will produce NaN)
                edge_tensor = torch::tensor({{1.5f, -1.5f}, {2.0f, -2.0f}}, options);
            } else if (edge_case_selector < 192) {
                // Values inside domain
                edge_tensor = torch::tensor({{0.0f, 0.5f}, {-0.5f, 0.707f}}, options);
            } else {
                // Zero tensor
                edge_tensor = torch::zeros({3, 3}, options);
            }
            
            // Apply arccos_ in-place
            edge_tensor.arccos_();
        }
        
        // Test with different dtypes if we have more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            
            try {
                torch::Tensor typed_tensor;
                if (dtype_selector < 128) {
                    typed_tensor = torch::rand({2, 3}, torch::TensorOptions().dtype(torch::kFloat64)) * 2.0 - 1.0;
                } else {
                    typed_tensor = torch::rand({3, 2}, torch::TensorOptions().dtype(torch::kFloat32)) * 2.0 - 1.0;
                }
                typed_tensor.arccos_();
            } catch (...) {
                // Silently handle expected failures for unsupported dtypes
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}