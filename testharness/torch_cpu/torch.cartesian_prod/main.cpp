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
        size_t offset = 0;
        
        // Determine number of tensors to create (1-4)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 4) + 1;
        
        // Create input tensors (must be 1D for cartesian_prod)
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) break;
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                // Flatten to 1D - cartesian_prod requires 1D tensors
                // Limit size to avoid combinatorial explosion
                tensor = tensor.flatten();
                if (tensor.numel() > 10) {
                    tensor = tensor.slice(0, 0, 10);
                }
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If one tensor creation fails, continue with the ones we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) return 0;
        
        // Apply cartesian_prod operation
        // Note: cartesian_prod works with 1 or more tensors
        torch::Tensor result = torch::cartesian_prod(tensors);
        
        // Perform operations on the result to ensure it's valid and increase coverage
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            auto dtype = result.dtype();
            auto dim = result.dim();
            
            // Try to access elements if tensor is not empty
            if (numel > 0) {
                auto first_elem = result.index({0});
                
                // Additional operations for coverage
                auto sum = result.sum();
                auto mean_result = result.to(torch::kFloat).mean();
                
                // Check shape is correct: should be (product of input sizes, num_tensors)
                if (dim == 2) {
                    auto num_rows = result.size(0);
                    auto num_cols = result.size(1);
                }
            }
            
            // Test contiguous and clone
            auto contiguous = result.contiguous();
            auto cloned = result.clone();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}