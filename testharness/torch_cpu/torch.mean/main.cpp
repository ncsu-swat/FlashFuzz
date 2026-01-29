#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Get tensor dimensions for safe dim selection
        int64_t ndim = input_tensor.dim();
        
        // Extract dimension parameter if we have more data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset < Size) {
            // Safely extract dimension - use modulo to keep within valid range
            if (ndim > 0) {
                dim = static_cast<int64_t>(Data[offset] % ndim);
                // Support negative indexing too
                if (offset + 1 < Size && (Data[offset + 1] & 0x1)) {
                    dim = -(ndim - dim);
                }
            }
            offset++;
            
            if (offset < Size) {
                offset++; // skip the negative dim selector byte
            }
            
            // Get keepdim parameter if available
            if (offset < Size) {
                keepdim = Data[offset++] & 0x1;
            }
        }
        
        // Apply torch.mean in different ways to test various code paths
        torch::Tensor result;
        
        // Test different variants of mean
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 4;
            
            switch (variant) {
                case 0:
                    // Mean over all dimensions
                    result = torch::mean(input_tensor);
                    break;
                    
                case 1:
                    // Mean over specific dimension
                    if (ndim > 0) {
                        try {
                            result = torch::mean(input_tensor, dim, keepdim);
                        } catch (...) {
                            // Shape/dimension errors are expected for some inputs
                            result = torch::mean(input_tensor);
                        }
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                    
                case 2:
                    // Mean with dtype specified
                    if (offset < Size) {
                        auto dtype_selector = Data[offset++];
                        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                        try {
                            result = torch::mean(input_tensor, dtype);
                        } catch (...) {
                            // dtype conversion might fail for some types
                            result = torch::mean(input_tensor);
                        }
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                    
                case 3:
                    // Mean with dimension and dtype
                    if (ndim > 0 && offset < Size) {
                        auto dtype_selector = Data[offset++];
                        auto dtype = fuzzer_utils::parseDataType(dtype_selector);
                        try {
                            result = torch::mean(input_tensor, dim, keepdim, dtype);
                        } catch (...) {
                            // Fallback on shape/dtype errors
                            result = torch::mean(input_tensor, dim, keepdim);
                        }
                    } else if (ndim > 0) {
                        result = torch::mean(input_tensor, dim, keepdim);
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
            }
        } else {
            // Default case if no variant byte available
            result = torch::mean(input_tensor);
        }
        
        // Access result to ensure computation is performed
        if (result.defined()) {
            auto numel = result.numel();
            if (numel > 0) {
                // Use sum instead of item() since mean can produce multi-element output with keepdim
                volatile float check = result.sum().item<float>();
                (void)check;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}