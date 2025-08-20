#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Determine number of tensors to create (1-5)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 5) + 1;
        
        // Create input tensors
        std::vector<torch::Tensor> tensors;
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) break;
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If one tensor creation fails, continue with the ones we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) return 0;
        
        // Apply cartesian_prod operation
        torch::Tensor result;
        if (tensors.size() == 1) {
            // Special case: cartesian product of a single tensor is the tensor itself
            result = tensors[0];
        } else {
            result = torch::cartesian_prod(tensors);
        }
        
        // Perform some basic operations on the result to ensure it's valid
        if (result.defined()) {
            auto sizes = result.sizes();
            auto numel = result.numel();
            auto dtype = result.dtype();
            
            // Try to access elements if tensor is not empty
            if (numel > 0) {
                auto first_elem = result.index({0});
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