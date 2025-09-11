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
        
        // Need at least a few bytes to create a tensor and specify dimensions
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dim parameter from the remaining data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Apply logsumexp operation
        torch::Tensor result;
        
        // If tensor has no dimensions, apply without dim parameter
        if (input.dim() == 0) {
            std::vector<int64_t> empty_dims;
            result = torch::logsumexp(input, empty_dims, keepdim);
        } else {
            // Ensure dim is within valid range for the tensor
            dim = dim % (2 * input.dim()) - input.dim();
            
            // Apply logsumexp with specified dim
            result = torch::logsumexp(input, dim, keepdim);
        }
        
        // Try with multiple dimensions if tensor has enough dimensions
        if (input.dim() >= 2 && offset < Size) {
            std::vector<int64_t> dims;
            int num_dims = Data[offset++] % input.dim() + 1;
            
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                    dims.push_back(d);
                }
            }
            
            if (!dims.empty()) {
                torch::Tensor result_multi = torch::logsumexp(input, dims, keepdim);
            }
        }
        
        // Try with named dimensions if available
        if (input.has_names() && input.dim() > 0) {
            torch::Tensor result_named = torch::logsumexp(input, input.names()[0], keepdim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
