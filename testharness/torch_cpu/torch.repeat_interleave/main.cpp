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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor - keep it small to avoid memory issues
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Limit input tensor size to avoid memory explosion with repeat_interleave
        if (input_tensor.numel() > 1000) {
            input_tensor = input_tensor.flatten().slice(0, 0, 100);
        }
        
        if (offset + 3 > Size) {
            return 0;
        }
        
        // Get control bytes for parameter selection
        uint8_t control1 = Data[offset++];
        uint8_t control2 = Data[offset++];
        uint8_t control3 = Data[offset++];
        
        // Determine which variant to test
        int variant = control1 % 4;
        
        torch::Tensor result;
        
        try {
            switch (variant) {
                case 0: {
                    // torch::repeat_interleave(Tensor repeats) - returns repeated indices
                    // Create a 1D tensor of repeat counts
                    int64_t num_repeats = (control2 % 5) + 1; // 1-5 elements
                    std::vector<int64_t> repeat_values;
                    for (int64_t i = 0; i < num_repeats && offset < Size; i++) {
                        // Keep repeat values small (0-5) to avoid memory explosion
                        repeat_values.push_back(Data[offset++] % 6);
                    }
                    if (repeat_values.empty()) {
                        repeat_values.push_back(1);
                    }
                    torch::Tensor repeats = torch::tensor(repeat_values, torch::kLong);
                    result = torch::repeat_interleave(repeats);
                    break;
                }
                case 1: {
                    // torch::repeat_interleave(Tensor self, int64_t repeats, dim)
                    int64_t repeats_int = (control2 % 5) + 1; // 1-5 repeats
                    int64_t ndim = input_tensor.dim();
                    if (ndim > 0) {
                        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(control3)) % ndim;
                        result = torch::repeat_interleave(input_tensor, repeats_int, dim);
                    } else {
                        // Scalar tensor - flatten and repeat
                        result = torch::repeat_interleave(input_tensor.flatten(), repeats_int, 0);
                    }
                    break;
                }
                case 2: {
                    // torch::repeat_interleave(Tensor self, Tensor repeats, dim)
                    int64_t ndim = input_tensor.dim();
                    if (ndim > 0) {
                        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(control3)) % ndim;
                        int64_t dim_size = input_tensor.size(dim);
                        // Create repeats tensor matching the dimension size
                        std::vector<int64_t> repeat_values;
                        for (int64_t i = 0; i < dim_size && i < 100; i++) {
                            int64_t rep_val = (offset < Size) ? (Data[offset++] % 4) : 1;
                            repeat_values.push_back(rep_val);
                        }
                        if (repeat_values.empty()) {
                            repeat_values.push_back(1);
                        }
                        torch::Tensor repeats = torch::tensor(repeat_values, torch::kLong);
                        result = torch::repeat_interleave(input_tensor, repeats, dim);
                    } else {
                        result = torch::repeat_interleave(input_tensor.unsqueeze(0), 2, 0);
                    }
                    break;
                }
                case 3: {
                    // torch::repeat_interleave with optional output_size
                    int64_t repeats_int = (control2 % 4) + 1;
                    int64_t ndim = input_tensor.dim();
                    if (ndim > 0) {
                        int64_t dim = 0;
                        int64_t dim_size = input_tensor.size(dim);
                        int64_t output_size = dim_size * repeats_int;
                        // Vary output_size slightly based on fuzzer input
                        if (control3 % 3 == 0 && output_size > 0) {
                            output_size = output_size - (control3 % output_size);
                        }
                        result = torch::repeat_interleave(input_tensor, repeats_int, dim, 
                            c10::optional<int64_t>(output_size));
                    } else {
                        result = torch::repeat_interleave(input_tensor.flatten(), repeats_int, 0);
                    }
                    break;
                }
            }
            
            // Use the result to ensure computation happens
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid dimensions, shapes, etc.
            // Silently ignore to continue fuzzing
        }
        catch (const std::runtime_error &e) {
            // Expected runtime errors
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}