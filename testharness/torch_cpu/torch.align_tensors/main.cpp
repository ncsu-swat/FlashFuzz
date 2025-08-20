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
        
        // Create a vector of tensors
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        // Create tensors with different properties
        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(tensor);
            } catch (const std::exception& e) {
                // If creating one tensor fails, continue with the ones we have
                break;
            }
        }
        
        // If we have at least one tensor, proceed with align_tensors
        if (!tensors.empty()) {
            // Apply torch.align_tensors operation
            std::vector<torch::Tensor> aligned_tensors = torch::align_tensors(tensors);
            
            // Verify the result
            if (aligned_tensors.size() != tensors.size()) {
                throw std::runtime_error("align_tensors returned different number of tensors");
            }
            
            // Check if all aligned tensors have the same dimensions
            if (aligned_tensors.size() > 1) {
                for (size_t i = 1; i < aligned_tensors.size(); ++i) {
                    if (aligned_tensors[i].dim() != aligned_tensors[0].dim()) {
                        throw std::runtime_error("aligned tensors have different dimensions");
                    }
                }
            }
            
            // Try to use the aligned tensors
            for (auto& tensor : aligned_tensors) {
                // Simple operation to ensure tensor is valid
                auto sum = tensor.sum();
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