#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for std operation if we have more data
        bool unbiased = false;
        bool keepdim = false;
        
        if (offset + 1 < Size) {
            unbiased = Data[offset++] & 0x1;
        }
        
        if (offset + 1 < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Try different variants of torch::std
        
        // Variant 1: std over all dimensions
        torch::Tensor result1 = torch::std(input_tensor, unbiased);
        
        // Variant 2: std with keepdim option
        torch::Tensor result2 = torch::std(input_tensor, unbiased, keepdim);
        
        // Variant 3: std along specific dimension if tensor has dimensions
        if (input_tensor.dim() > 0 && offset < Size) {
            // Get a dimension to reduce along
            int64_t dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), input_tensor.dim());
            
            // Try negative dimension index too
            if (offset < Size && (Data[offset++] & 0x1)) {
                dim = -dim - 1;
            }
            
            // std along dimension
            torch::Tensor result3 = torch::std(input_tensor, dim, unbiased);
            
            // std along dimension with keepdim
            torch::Tensor result4 = torch::std(input_tensor, dim, unbiased, keepdim);
            
            // Try with a list of dimensions if tensor has multiple dimensions
            if (input_tensor.dim() > 1 && offset < Size) {
                std::vector<int64_t> dims;
                uint8_t num_dims = Data[offset++] % input_tensor.dim();
                
                for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                    
                    // Ensure no duplicate dimensions
                    if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                        dims.push_back(d);
                    }
                }
                
                if (!dims.empty()) {
                    // std along multiple dimensions
                    torch::Tensor result5 = torch::std(input_tensor, dims, unbiased);
                    
                    // std along multiple dimensions with keepdim
                    torch::Tensor result6 = torch::std(input_tensor, dims, unbiased, keepdim);
                }
            }
        }
        
        // Try named dimension variant if we have more data
        if (offset < Size && input_tensor.dim() > 0) {
            // Create a named tensor by adding names to dimensions
            std::vector<torch::Dimname> names;
            for (int i = 0; i < input_tensor.dim(); i++) {
                names.push_back(torch::Dimname::fromSymbol(torch::Symbol::dimname(std::to_string(i))));
            }
            
            auto named_tensor = input_tensor.refine_names(names);
            
            // Choose a dimension name to reduce along
            int64_t dim_idx = Data[offset++] % named_tensor.dim();
            auto dim_name = names[dim_idx];
            
            // std with named dimension
            torch::Tensor result7 = torch::std(named_tensor, dim_name, unbiased);
            
            // std with named dimension and keepdim
            torch::Tensor result8 = torch::std(named_tensor, dim_name, unbiased, keepdim);
        }
        
        // Try correction parameter variant (available in newer PyTorch versions)
        if (offset < Size) {
            double correction = static_cast<double>(Data[offset++]) / 255.0;
            
            // std with correction parameter - use explicit dimension list
            std::vector<int64_t> empty_dims;
            torch::Tensor result9 = torch::std(input_tensor, empty_dims, keepdim, correction);
            
            // If tensor has dimensions, try with specific dimension
            if (input_tensor.dim() > 0) {
                int64_t dim = offset < Size ? 
                    (static_cast<int64_t>(Data[offset++]) % input_tensor.dim()) : 0;
                
                torch::Tensor result10 = torch::std(input_tensor, dim, keepdim, correction);
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