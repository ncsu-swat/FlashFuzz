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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a dimension value from the data if available
        int64_t dim = -1;
        bool has_dim = false;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            has_dim = true;
        }
        
        // Extract a keepdim boolean from the data if available
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Apply torch.any in different ways
        torch::Tensor result;
        
        // Test case 1: torch.any without arguments
        result = torch::any(input_tensor);
        
        // Test case 2: torch.any with dimension and keepdim
        if (has_dim && input_tensor.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            
            result = torch::any(input_tensor, dim, keepdim);
        }
        
        // Test case 3: torch.any with dimension only
        if (has_dim && input_tensor.dim() > 0) {
            result = torch::any(input_tensor, dim);
        }
        
        // Test case 4: torch.any with named dimension
        if (input_tensor.dim() > 0) {
            // Create a named tensor if possible
            std::vector<torch::Dimname> names;
            for (int i = 0; i < input_tensor.dim(); i++) {
                names.push_back(torch::Dimname::wildcard());
            }
            
            auto named_tensor = input_tensor.refine_names(names);
            
            if (has_dim && dim >= 0 && dim < named_tensor.dim()) {
                result = torch::any(named_tensor, named_tensor.names()[dim], keepdim);
            }
        }
        
        // Test case 5: torch.any with multiple dimensions
        if (input_tensor.dim() >= 2) {
            std::vector<int64_t> dims;
            
            // Extract up to 2 dimensions
            int64_t dim1 = 0;
            int64_t dim2 = 1;
            
            if (has_dim) {
                dim1 = dim % input_tensor.dim();
                if (dim1 < 0) dim1 += input_tensor.dim();
                
                dim2 = (dim1 + 1) % input_tensor.dim();
            }
            
            dims.push_back(dim1);
            dims.push_back(dim2);
            
            result = torch::any(input_tensor, dims, keepdim);
        }
        
        // Test case 6: torch.any with out parameter using any_out
        if (input_tensor.dim() > 0 && has_dim) {
            dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
            
            torch::Tensor out = torch::empty({}, torch::kBool);
            result = torch::any_out(out, input_tensor, dim, keepdim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
