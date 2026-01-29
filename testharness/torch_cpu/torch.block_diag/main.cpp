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
        
        // Determine number of tensors to create (1-8)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 8) + 1;
        
        // Create a vector to store the tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
        for (uint8_t i = 0; i < num_tensors; i++) {
            if (offset >= Size) break;
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If tensor creation fails, just continue with the tensors we have
                break;
            }
        }
        
        // If we have at least one tensor, apply block_diag
        if (!tensors.empty()) {
            torch::Tensor result = torch::block_diag(tensors);
            
            // Perform some operation on the result to ensure it's used
            auto sum = result.sum();
            
            // Test edge cases by creating additional block_diag calls with subsets of tensors
            if (tensors.size() > 1) {
                // Test with first tensor only
                torch::Tensor single_result = torch::block_diag({tensors[0]});
                
                // Test with last two tensors if available
                if (tensors.size() > 2) {
                    std::vector<torch::Tensor> subset_tensors = {
                        tensors[tensors.size()-2], 
                        tensors[tensors.size()-1]
                    };
                    torch::Tensor subset_result = torch::block_diag(subset_tensors);
                }
            }
            
            // Test with empty vector edge case (should produce empty tensor)
            try {
                std::vector<torch::Tensor> empty_tensors;
                torch::Tensor empty_result = torch::block_diag(empty_tensors);
            } catch (...) {
                // Empty input might not be supported, silently ignore
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