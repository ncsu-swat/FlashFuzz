#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and properties
        auto dims = parse_tensor_dims(Data, Size, offset);
        if (dims.empty()) return 0;

        auto dtype = parse_dtype(Data, Size, offset);
        if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
            return 0; // nanmedian doesn't support complex types
        }

        // Create input tensor with potential NaN values
        auto input = create_tensor(dims, dtype, Data, Size, offset);
        if (!input.defined()) return 0;

        // Inject some NaN values to test nanmedian behavior
        if (input.numel() > 0 && input.dtype().isFloatingPoint()) {
            auto flat = input.flatten();
            int64_t num_nans = std::min(static_cast<int64_t>(3), flat.numel());
            for (int64_t i = 0; i < num_nans && offset < Size; ++i) {
                int64_t idx = static_cast<int64_t>(Data[offset % Size]) % flat.numel();
                flat[idx] = std::numeric_limits<double>::quiet_NaN();
                offset++;
            }
        }

        // Test 1: nanmedian without dimension (returns scalar)
        auto result1 = torch::nanmedian(input);

        // Test 2: nanmedian with dimension
        if (input.dim() > 0 && offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset % Size]) % input.dim();
            offset++;
            
            bool keepdim = offset < Size ? (Data[offset % Size] % 2 == 0) : false;
            offset++;

            // Test with dimension specified
            auto result2 = torch::nanmedian(input, dim, keepdim);
            
            // Verify result is a tuple (values, indices)
            auto values = std::get<0>(result2);
            auto indices = std::get<1>(result2);
            
            // Basic sanity checks
            if (values.defined() && indices.defined()) {
                // Check that indices are valid
                if (indices.numel() > 0) {
                    auto max_idx = torch::max(indices);
                    auto min_idx = torch::min(indices);
                    if (max_idx.item<int64_t>() >= input.size(dim) || 
                        min_idx.item<int64_t>() < 0) {
                        std::cerr << "Invalid indices in nanmedian result" << std::endl;
                    }
                }
            }
        }

        // Test 3: Edge cases with different tensor shapes
        if (input.numel() > 0) {
            // Test with all NaN tensor
            auto all_nan_tensor = torch::full_like(input, std::numeric_limits<double>::quiet_NaN());
            if (all_nan_tensor.dtype().isFloatingPoint()) {
                auto nan_result = torch::nanmedian(all_nan_tensor);
            }

            // Test with single element tensor
            if (input.numel() >= 1) {
                auto single_elem = input.flatten().slice(0, 0, 1).reshape({1});
                auto single_result = torch::nanmedian(single_elem);
            }
        }

        // Test 4: Different data types
        if (input.dtype() != torch::kFloat32 && input.dtype().isFloatingPoint()) {
            auto float_input = input.to(torch::kFloat32);
            auto float_result = torch::nanmedian(float_input);
        }

        // Test 5: Empty tensor handling
        if (offset < Size && Data[offset % Size] % 10 == 0) {
            try {
                auto empty_tensor = torch::empty({0}, input.options());
                auto empty_result = torch::nanmedian(empty_tensor);
            } catch (...) {
                // Empty tensor might throw, which is acceptable
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}