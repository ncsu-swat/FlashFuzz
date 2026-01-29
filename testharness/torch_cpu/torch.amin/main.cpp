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
        
        // Extract mode selector
        uint8_t mode = 0;
        if (offset < Size) {
            mode = Data[offset] % 4;
            offset++;
        }
        
        // Extract dim parameter if we have more data
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter if we have more data
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        torch::Tensor result;
        
        switch (mode) {
            case 0: {
                // Case 1: No dimension specified (reduce over all dimensions)
                result = torch::amin(input_tensor);
                break;
            }
            case 1: {
                // Case 2: Specific dimension with keepdim option
                // Normalize dim to valid range to exercise the API more effectively
                if (input_tensor.dim() > 0) {
                    int64_t normalized_dim = dim % input_tensor.dim();
                    if (normalized_dim < 0) {
                        normalized_dim += input_tensor.dim();
                    }
                    result = torch::amin(input_tensor, normalized_dim, keepdim);
                } else {
                    // 0-dim tensor
                    result = torch::amin(input_tensor);
                }
                break;
            }
            case 2: {
                // Case 3: Test with potentially invalid dim (let PyTorch validate)
                try {
                    result = torch::amin(input_tensor, dim, keepdim);
                } catch (const std::exception &) {
                    // Expected for invalid dimensions - silent catch
                    result = torch::amin(input_tensor);
                }
                break;
            }
            case 3: {
                // Case 4: Multiple dimensions
                if (input_tensor.dim() >= 2 && offset < Size) {
                    std::vector<int64_t> dims;
                    
                    // Extract number of dimensions to reduce over (1 to min(ndim, 3))
                    int64_t num_dims = 1 + (Data[offset] % std::min(input_tensor.dim(), static_cast<int64_t>(3)));
                    offset++;
                    
                    // Generate unique dimension indices
                    for (int64_t i = 0; i < num_dims && i < input_tensor.dim(); i++) {
                        int64_t d = i;
                        if (offset < Size) {
                            d = Data[offset] % input_tensor.dim();
                            offset++;
                        }
                        // Check for duplicates
                        bool duplicate = false;
                        for (const auto& existing : dims) {
                            if (existing == d || existing == d - input_tensor.dim() || 
                                existing + input_tensor.dim() == d) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (!duplicate) {
                            dims.push_back(d);
                        }
                    }
                    
                    // Apply amin with multiple dimensions
                    if (!dims.empty()) {
                        try {
                            result = torch::amin(input_tensor, dims, keepdim);
                        } catch (const std::exception &) {
                            // Expected for invalid dimension combinations - silent catch
                            result = torch::amin(input_tensor);
                        }
                    } else {
                        result = torch::amin(input_tensor);
                    }
                } else {
                    result = torch::amin(input_tensor);
                }
                break;
            }
        }
        
        // Ensure the result is used to prevent optimization
        if (result.defined() && result.numel() > 0) {
            try {
                volatile float dummy = result.sum().item<float>();
                (void)dummy;
            } catch (const std::exception &) {
                // Silent catch for dtype conversion issues
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}