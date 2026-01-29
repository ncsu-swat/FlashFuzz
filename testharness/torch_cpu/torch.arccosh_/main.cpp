#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor from fuzzer data
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test 1: Basic in-place arccosh_ operation
        {
            torch::Tensor input_copy = input.clone();
            input_copy.arccosh_();
        }
        
        // Test 2: Try with different dtypes if we have more data
        if (offset < Size) {
            torch::Tensor float_input = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat32);
            float_input.arccosh_();
        }
        
        if (offset < Size) {
            torch::Tensor double_input = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat64);
            double_input.arccosh_();
        }
        
        // Test 3: Test with specific edge case values
        // arccosh is defined for x >= 1 (returns NaN for x < 1 with real tensors)
        {
            torch::Tensor edge_case = torch::tensor({1.0, 1.5, 2.0, 10.0, 100.0});
            edge_case.arccosh_();
        }
        
        // Test 4: Empty tensor (edge case)
        {
            torch::Tensor empty_tensor = torch::empty({0}, torch::kFloat32);
            empty_tensor.arccosh_();
        }
        
        // Test 5: Scalar tensor
        {
            torch::Tensor scalar_tensor = torch::tensor(2.0);
            scalar_tensor.arccosh_();
        }
        
        // Test 6: Multi-dimensional tensor
        if (offset < Size) {
            torch::Tensor multi_dim = fuzzer_utils::createTensor(Data, Size, offset);
            if (multi_dim.numel() >= 4) {
                try {
                    // Reshape to 2D if possible
                    int64_t numel = multi_dim.numel();
                    int64_t rows = 2;
                    int64_t cols = numel / rows;
                    if (rows * cols == numel) {
                        torch::Tensor reshaped = multi_dim.reshape({rows, cols});
                        reshaped.arccosh_();
                    }
                } catch (...) {
                    // Shape manipulation may fail, continue
                }
            }
        }
        
        // Test 7: Contiguous vs non-contiguous tensor
        if (offset < Size) {
            torch::Tensor base = fuzzer_utils::createTensor(Data, Size, offset);
            if (base.numel() >= 4) {
                try {
                    torch::Tensor transposed = base.reshape({2, -1}).t().clone();
                    transposed.arccosh_();
                } catch (...) {
                    // Reshape/transpose may fail, continue
                }
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