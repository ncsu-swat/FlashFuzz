#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor shape and data type
        auto shape = parse_tensor_shape(Data, Size, offset);
        if (shape.empty()) return 0;
        
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Create input tensor with potential NaN values
        torch::Tensor input = create_tensor_with_data(Data, Size, offset, shape, dtype);
        
        // Test basic nanmean without dimensions
        torch::Tensor result1 = torch::nanmean(input);
        
        // Parse dimension parameter for dimensional reduction
        if (offset < Size) {
            int64_t dim = parse_int_in_range(Data, Size, offset, -static_cast<int64_t>(input.dim()), input.dim() - 1);
            
            // Test nanmean with specific dimension
            torch::Tensor result2 = torch::nanmean(input, dim);
            
            // Test nanmean with keepdim=true
            torch::Tensor result3 = torch::nanmean(input, dim, true);
            
            // Test nanmean with keepdim=false
            torch::Tensor result4 = torch::nanmean(input, dim, false);
        }
        
        // Test with multiple dimensions if tensor has enough dimensions
        if (input.dim() >= 2 && offset < Size) {
            std::vector<int64_t> dims;
            int num_dims = parse_int_in_range(Data, Size, offset, 1, std::min(3, static_cast<int>(input.dim())));
            
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int64_t dim = parse_int_in_range(Data, Size, offset, -static_cast<int64_t>(input.dim()), input.dim() - 1);
                dims.push_back(dim);
            }
            
            if (!dims.empty()) {
                torch::Tensor result5 = torch::nanmean(input, dims);
                torch::Tensor result6 = torch::nanmean(input, dims, true);
            }
        }
        
        // Test with different data types by casting
        if (offset < Size) {
            auto target_dtype = parse_dtype(Data, Size, offset);
            if (target_dtype != input.dtype()) {
                try {
                    torch::Tensor casted = input.to(target_dtype);
                    torch::Tensor result7 = torch::nanmean(casted);
                } catch (...) {
                    // Ignore casting errors
                }
            }
        }
        
        // Test edge cases with special tensors
        if (offset < Size) {
            uint8_t test_case = Data[offset++] % 4;
            
            switch (test_case) {
                case 0: {
                    // Test with all NaN tensor
                    torch::Tensor nan_tensor = torch::full_like(input, std::numeric_limits<double>::quiet_NaN());
                    torch::Tensor result8 = torch::nanmean(nan_tensor);
                    break;
                }
                case 1: {
                    // Test with mixed NaN and finite values
                    torch::Tensor mixed = input.clone();
                    if (mixed.numel() > 0) {
                        mixed.flatten()[0] = std::numeric_limits<double>::quiet_NaN();
                        torch::Tensor result9 = torch::nanmean(mixed);
                    }
                    break;
                }
                case 2: {
                    // Test with infinity values
                    if (input.dtype().is_floating_point()) {
                        torch::Tensor inf_tensor = input.clone();
                        if (inf_tensor.numel() > 0) {
                            inf_tensor.flatten()[0] = std::numeric_limits<double>::infinity();
                            torch::Tensor result10 = torch::nanmean(inf_tensor);
                        }
                    }
                    break;
                }
                case 3: {
                    // Test with very large/small values
                    if (input.dtype().is_floating_point()) {
                        torch::Tensor extreme = input.clone();
                        if (extreme.numel() > 1) {
                            extreme.flatten()[0] = std::numeric_limits<double>::max();
                            extreme.flatten()[1] = std::numeric_limits<double>::lowest();
                            torch::Tensor result11 = torch::nanmean(extreme);
                        }
                    }
                    break;
                }
            }
        }
        
        // Test with empty tensor
        if (offset < Size && Data[offset++] % 10 == 0) {
            torch::Tensor empty_tensor = torch::empty({0}, input.options());
            try {
                torch::Tensor result12 = torch::nanmean(empty_tensor);
            } catch (...) {
                // Expected to potentially fail
            }
        }
        
        // Test with scalar tensor
        if (offset < Size && Data[offset++] % 8 == 0) {
            torch::Tensor scalar = torch::scalar_tensor(parse_float(Data, Size, offset), input.options());
            torch::Tensor result13 = torch::nanmean(scalar);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}