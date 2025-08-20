#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Determine number of tensors to create (1-4)
        uint8_t num_tensors = (Size > 0) ? (Data[0] % 4) + 1 : 1;
        offset++;
        
        // Create a vector to hold our tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If we can't create a tensor, just continue with what we have
                break;
            }
        }
        
        // Need at least one tensor to proceed
        if (tensors.empty()) {
            return 0;
        }
        
        // Apply vstack operation
        torch::Tensor result = torch::vstack(tensors);
        
        // Optional: Test some properties of the result
        auto result_sizes = result.sizes();
        int64_t expected_first_dim = 0;
        for (const auto& t : tensors) {
            if (t.dim() == 0) {
                expected_first_dim += 1;
            } else {
                expected_first_dim += t.size(0);
            }
        }
        
        // Use the result to ensure the operation is not optimized away
        volatile bool has_nan = result.isnan().any().item<bool>();
        volatile int64_t numel = result.numel();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}