#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 16) return 0;

        // Extract tensor dimensions and properties
        auto tensor_dims = extract_tensor_dims(Data, Size, offset, 1, 4); // 1-4 dimensions
        if (tensor_dims.empty()) return 0;

        auto dtype = extract_dtype(Data, Size, offset);
        auto device = extract_device(Data, Size, offset);

        // Create input tensor with random values
        torch::Tensor input = create_tensor(tensor_dims, dtype, device);
        if (!input.defined()) return 0;

        // Extract repeats parameter - can be int or tensor
        uint8_t repeats_type = extract_uint8(Data, Size, offset) % 3;
        
        if (repeats_type == 0) {
            // Test with integer repeats
            int64_t repeats = extract_int64(Data, Size, offset, 1, 10);
            
            // Test without dim parameter
            torch::Tensor result1 = torch::repeat_interleave(input, repeats);
            
            // Test with dim parameter
            if (input.dim() > 0) {
                int64_t dim = extract_int64(Data, Size, offset, 0, input.dim() - 1);
                torch::Tensor result2 = torch::repeat_interleave(input, repeats, dim);
                
                // Test with optional output_size
                int64_t output_size = extract_int64(Data, Size, offset, 1, 100);
                torch::Tensor result3 = torch::repeat_interleave(input, repeats, dim, output_size);
            }
        }
        else if (repeats_type == 1) {
            // Test with tensor repeats
            std::vector<int64_t> repeats_dims = {extract_int64(Data, Size, offset, 1, 20)};
            torch::Tensor repeats_tensor = create_tensor(repeats_dims, torch::kLong, device);
            
            // Ensure repeats tensor has positive values
            repeats_tensor = torch::abs(repeats_tensor) + 1;
            
            // Test without dim parameter
            torch::Tensor result1 = torch::repeat_interleave(input, repeats_tensor);
            
            // Test with dim parameter
            if (input.dim() > 0) {
                int64_t dim = extract_int64(Data, Size, offset, 0, input.dim() - 1);
                
                // Adjust repeats tensor size to match input dimension
                int64_t dim_size = input.size(dim);
                if (repeats_tensor.numel() != dim_size) {
                    repeats_tensor = repeats_tensor.narrow(0, 0, std::min(repeats_tensor.numel(), dim_size));
                    if (repeats_tensor.numel() < dim_size) {
                        repeats_tensor = torch::cat({repeats_tensor, torch::ones({dim_size - repeats_tensor.numel()}, torch::kLong)});
                    }
                }
                
                torch::Tensor result2 = torch::repeat_interleave(input, repeats_tensor, dim);
                
                // Test with optional output_size
                int64_t output_size = extract_int64(Data, Size, offset, 1, 100);
                torch::Tensor result3 = torch::repeat_interleave(input, repeats_tensor, dim, output_size);
            }
        }
        else {
            // Test edge cases
            
            // Test with zero repeats
            torch::Tensor result_zero = torch::repeat_interleave(input, 0);
            
            // Test with large repeats (but reasonable to avoid OOM)
            int64_t large_repeats = extract_int64(Data, Size, offset, 50, 100);
            if (input.numel() * large_repeats < 10000) { // Avoid OOM
                torch::Tensor result_large = torch::repeat_interleave(input, large_repeats);
            }
            
            // Test with negative dim (should wrap around)
            if (input.dim() > 0) {
                int64_t neg_dim = -extract_int64(Data, Size, offset, 1, input.dim());
                torch::Tensor result_neg_dim = torch::repeat_interleave(input, 2, neg_dim);
            }
            
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0}, dtype).to(device);
            torch::Tensor result_empty = torch::repeat_interleave(empty_tensor, 3);
            
            // Test with scalar tensor
            torch::Tensor scalar_tensor = torch::scalar_tensor(extract_float(Data, Size, offset), dtype).to(device);
            torch::Tensor result_scalar = torch::repeat_interleave(scalar_tensor, 5);
        }

        // Test the self variant (method call)
        int64_t method_repeats = extract_int64(Data, Size, offset, 1, 5);
        torch::Tensor method_result = input.repeat_interleave(method_repeats);
        
        if (input.dim() > 0) {
            int64_t method_dim = extract_int64(Data, Size, offset, 0, input.dim() - 1);
            torch::Tensor method_result_dim = input.repeat_interleave(method_repeats, method_dim);
        }

        // Test with different tensor types
        if (dtype != torch::kBool) {
            torch::Tensor bool_tensor = (input > 0).to(device);
            torch::Tensor bool_result = torch::repeat_interleave(bool_tensor, 2);
        }

        // Test with complex tensors if supported
        if (dtype.isFloatingPoint()) {
            torch::Tensor complex_tensor = torch::complex(input, input).to(device);
            torch::Tensor complex_result = torch::repeat_interleave(complex_tensor, 2);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}