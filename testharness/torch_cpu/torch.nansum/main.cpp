#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
        
        torch::Tensor result;
        
        // Case 1: nansum over all dimensions (reduces to scalar)
        result = torch::nansum(input_tensor);
        
        // Case 2: nansum with specified dimension
        if (input_tensor.dim() > 0) {
            // Ensure dim is within valid range for the tensor
            int64_t actual_dim = dim % input_tensor.dim();
            if (actual_dim < 0) actual_dim += input_tensor.dim();
            
            try {
                result = torch::nansum(input_tensor, actual_dim, keepdim);
            } catch (...) {
                // Shape/dim errors are expected for some inputs
            }
            
            try {
                // Case 3: nansum with dimension and explicit keepdim=false
                result = torch::nansum(input_tensor, actual_dim, false);
            } catch (...) {
            }
            
            try {
                // Case 4: nansum with dimension and explicit keepdim=true
                result = torch::nansum(input_tensor, actual_dim, true);
            } catch (...) {
            }
        }
        
        // Case 5: nansum with dimension array if tensor has multiple dimensions
        if (input_tensor.dim() > 1) {
            std::vector<int64_t> dims;
            
            // Create a list of dimensions to sum over
            for (int64_t i = 0; i < input_tensor.dim(); i++) {
                if ((i % 2) == 0) {
                    dims.push_back(i);
                }
            }
            
            if (!dims.empty()) {
                try {
                    result = torch::nansum(input_tensor, dims, keepdim);
                } catch (...) {
                    // Some dimension combinations may be invalid
                }
            }
        }
        
        // Case 6: nansum with dtype specified (must include dim parameter)
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Try with dim and dtype - nansum requires dim when specifying dtype
            if (input_tensor.dim() > 0) {
                int64_t actual_dim = dim % input_tensor.dim();
                if (actual_dim < 0) actual_dim += input_tensor.dim();
                
                try {
                    result = torch::nansum(input_tensor, actual_dim, keepdim, dtype);
                } catch (...) {
                    // dtype/dim combinations may fail
                }
            }
            
            // Try with IntArrayRef dims and dtype
            if (input_tensor.dim() > 1) {
                std::vector<int64_t> dims_vec;
                for (int64_t i = 0; i < input_tensor.dim(); i += 2) {
                    dims_vec.push_back(i);
                }
                if (!dims_vec.empty()) {
                    try {
                        result = torch::nansum(input_tensor, dims_vec, keepdim, dtype);
                    } catch (...) {
                        // dtype/dims combinations may fail
                    }
                }
            }
        }
        
        // Case 7: Test with tensors containing NaN values explicitly
        try {
            torch::Tensor nan_tensor = input_tensor.clone();
            if (nan_tensor.numel() > 0 && nan_tensor.is_floating_point()) {
                // Set some elements to NaN to exercise the nan-handling code path
                auto flat = nan_tensor.flatten();
                if (flat.numel() > 0) {
                    flat[0] = std::numeric_limits<float>::quiet_NaN();
                }
                result = torch::nansum(nan_tensor);
                
                // Also test with dim on nan tensor
                if (nan_tensor.dim() > 0) {
                    result = torch::nansum(nan_tensor, 0, keepdim);
                }
            }
        } catch (...) {
            // May fail for non-floating point tensors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}