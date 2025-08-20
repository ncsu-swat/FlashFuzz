#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Determine number of tensors to create (1-4)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 4) + 1;
        
        // Create a vector to store the tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors based on the input data
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If tensor creation fails, just continue with the tensors we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply broadcast_tensors operation
        try {
            std::vector<torch::Tensor> result = torch::broadcast_tensors(tensors);
            
            // Verify the result by checking some properties
            if (!result.empty()) {
                // All output tensors should have the same shape
                for (size_t i = 1; i < result.size(); ++i) {
                    if (result[0].sizes() != result[i].sizes()) {
                        throw std::runtime_error("Broadcast tensors produced inconsistent shapes");
                    }
                }
                
                // Perform some operations on the result to ensure it's valid
                for (auto& tensor : result) {
                    // Simple operation to check tensor validity
                    torch::Tensor sum = tensor.sum();
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected for invalid inputs
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}