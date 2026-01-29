#include "fuzzer_utils.h"
#include <iostream>
#include <set>

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
        
        // Create input tensor - nanmean requires floating point tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to floating point if not already
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Optionally introduce NaN values to actually test nanmean functionality
        if (offset < Size && (Data[offset++] & 0x1)) {
            // Create a mask for where to put NaNs
            auto mask = torch::rand_like(input_tensor) < 0.3;
            input_tensor = torch::where(mask, 
                torch::full_like(input_tensor, std::numeric_limits<float>::quiet_NaN()), 
                input_tensor);
        }
        
        // Extract control byte for which variant to test
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        // Extract keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset++] & 0x1);
        }
        
        torch::Tensor result;
        
        if (input_tensor.dim() == 0) {
            // Scalar tensor - only global nanmean works
            result = torch::nanmean(input_tensor);
        } else {
            switch (variant) {
                case 0: {
                    // nanmean over all dimensions (no dim argument)
                    result = torch::nanmean(input_tensor);
                    break;
                }
                case 1: {
                    // nanmean along a single dimension
                    int64_t dim = 0;
                    if (offset + sizeof(int8_t) <= Size) {
                        int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                        dim = dim_byte % input_tensor.dim();
                    }
                    
                    try {
                        result = torch::nanmean(input_tensor, dim, keepdim);
                    } catch (const std::exception &) {
                        // Silently handle expected failures
                        result = torch::nanmean(input_tensor);
                    }
                    break;
                }
                case 2: {
                    // nanmean with IntArrayRef (multiple dimensions)
                    std::set<int64_t> unique_dims;
                    int num_dims = std::min(static_cast<int>(Size - offset), std::min(4, static_cast<int>(input_tensor.dim())));
                    
                    for (int i = 0; i < num_dims && offset < Size; i++) {
                        int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                        int64_t d = dim_byte % input_tensor.dim();
                        // Normalize negative dims
                        if (d < 0) {
                            d += input_tensor.dim();
                        }
                        unique_dims.insert(d);
                    }
                    
                    if (!unique_dims.empty()) {
                        std::vector<int64_t> dims(unique_dims.begin(), unique_dims.end());
                        try {
                            result = torch::nanmean(input_tensor, dims, keepdim);
                        } catch (const std::exception &) {
                            // Silently handle expected failures
                            result = torch::nanmean(input_tensor);
                        }
                    } else {
                        result = torch::nanmean(input_tensor);
                    }
                    break;
                }
                case 3: {
                    // nanmean with output tensor
                    int64_t dim = 0;
                    if (offset + sizeof(int8_t) <= Size) {
                        int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                        dim = dim_byte % input_tensor.dim();
                    }
                    
                    try {
                        // Compute expected output shape
                        auto sizes = input_tensor.sizes().vec();
                        if (keepdim) {
                            sizes[dim >= 0 ? dim : dim + input_tensor.dim()] = 1;
                        } else {
                            sizes.erase(sizes.begin() + (dim >= 0 ? dim : dim + input_tensor.dim()));
                        }
                        
                        torch::Tensor out = torch::empty(sizes, input_tensor.options());
                        result = torch::nanmean_out(out, input_tensor, dim, keepdim);
                    } catch (const std::exception &) {
                        // Silently handle expected failures
                        result = torch::nanmean(input_tensor);
                    }
                    break;
                }
                default:
                    result = torch::nanmean(input_tensor);
                    break;
            }
        }
        
        // Verify the result is a valid tensor
        if (!result.defined()) {
            throw std::runtime_error("nanmean returned undefined tensor");
        }
        
        // Force computation to ensure any lazy evaluation is triggered
        (void)result.numel();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}