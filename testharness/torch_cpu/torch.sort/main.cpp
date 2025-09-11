#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to sort along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get descending flag
        bool descending = false;
        if (offset < Size) {
            descending = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply torch.sort operation
        if (input_tensor.dim() > 0) {
            // Normalize dim to be within valid range
            dim = dim % input_tensor.dim();
            if (dim < 0) {
                dim += input_tensor.dim();
            }
            
            // Call sort
            auto result = torch::sort(input_tensor, dim, descending);
            
            // Access the values and indices to ensure they're computed
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
            
            // Test the stable sort variant
            auto stable_result = torch::sort(input_tensor, dim, descending, true);
            torch::Tensor stable_values = std::get<0>(stable_result);
            torch::Tensor stable_indices = std::get<1>(stable_result);
        } else {
            // For 0-dim tensors, sort without dimension
            auto result = torch::sort(input_tensor);
            torch::Tensor values = std::get<0>(result);
            torch::Tensor indices = std::get<1>(result);
        }
        
        // Try sorting with named dimension
        if (input_tensor.dim() > 0) {
            // Create a named tensor if possible
            std::vector<torch::Dimname> names;
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                std::string name_str = "dim" + std::to_string(i);
                names.push_back(torch::Dimname::fromSymbol(c10::Symbol::dimname(name_str)));
            }
            
            auto named_tensor = input_tensor.refine_names(names);
            
            // Sort using a named dimension
            auto named_result = torch::sort(named_tensor, names[dim % names.size()], descending);
            torch::Tensor named_values = std::get<0>(named_result);
            torch::Tensor named_indices = std::get<1>(named_result);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
