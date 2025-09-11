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
        
        // Extract dim parameter if we have more data
        int64_t dim = -1;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Apply nanmean operation in different ways based on available data
        torch::Tensor result;
        
        // Test different variants of nanmean
        if (dim >= -input_tensor.dim() && dim < input_tensor.dim()) {
            // Case 1: nanmean along specific dimension
            result = torch::nanmean(input_tensor, dim, keepdim);
        } else if (dim == -1) {
            // Case 2: nanmean over all dimensions
            result = torch::nanmean(input_tensor);
        } else {
            // Case 3: nanmean with dimension list
            std::vector<int64_t> dims;
            
            // Create a list of dimensions to reduce over
            int num_dims = std::min(static_cast<int>(Size - offset), 4);
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int64_t d = static_cast<int64_t>(Data[offset++]) % (input_tensor.dim() * 2) - input_tensor.dim();
                dims.push_back(d);
            }
            
            if (!dims.empty()) {
                result = torch::nanmean(input_tensor, dims, keepdim);
            } else {
                result = torch::nanmean(input_tensor);
            }
        }
        
        // Verify the result is a valid tensor
        if (!result.defined()) {
            throw std::runtime_error("nanmean returned undefined tensor");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
