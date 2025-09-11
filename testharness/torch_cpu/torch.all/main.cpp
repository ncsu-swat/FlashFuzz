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
        
        // Extract a dimension to use for the all operation if there's enough data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
        }
        
        // Extract keepdim parameter if there's enough data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Test torch.all with different variants
        
        // Variant 1: all() - reduce over all dimensions
        torch::Tensor result1 = torch::all(input_tensor);
        
        // Variant 2: all(dim, keepdim) - reduce over specific dimension
        if (input_tensor.dim() > 0) {
            torch::Tensor result2 = torch::all(input_tensor, dim, keepdim);
        }
        
        // Variant 3: all(dim) - reduce over specific dimension without keepdim
        if (input_tensor.dim() > 0) {
            torch::Tensor result3 = torch::all(input_tensor, dim);
        }
        
        // Variant 4: Test with named dimension if tensor has names
        if (offset < Size && input_tensor.dim() > 0) {
            try {
                // Create a named tensor by adding names to dimensions
                std::vector<torch::Dimname> names;
                for (int i = 0; i < input_tensor.dim(); i++) {
                    names.push_back(torch::Dimname::wildcard());
                }
                auto named_tensor = input_tensor.refine_names(names);
                
                // Test all with Dimname
                torch::Tensor result4 = torch::all(named_tensor, names[dim % names.size()], keepdim);
            } catch (const std::exception &) {
                // Ignore exceptions from named tensor operations
            }
        }
        
        // Variant 5: Test with boolean tensor explicitly
        if (input_tensor.dtype() != torch::kBool) {
            // Convert to boolean tensor
            torch::Tensor bool_tensor = input_tensor.to(torch::kBool);
            torch::Tensor result5 = torch::all(bool_tensor);
            
            if (bool_tensor.dim() > 0) {
                torch::Tensor result6 = torch::all(bool_tensor, dim, keepdim);
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
