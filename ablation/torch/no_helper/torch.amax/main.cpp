#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and data types
        auto input_tensor = generate_tensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Get tensor dimensions
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            // For scalar tensors, test amax without dim parameter
            auto result = torch::amax(input_tensor);
            return 0;
        }

        // Test 1: Single dimension reduction
        if (offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset]) % ndim;
            offset++;
            
            bool keepdim = (offset < Size) ? (Data[offset] % 2 == 0) : false;
            offset++;

            auto result1 = torch::amax(input_tensor, dim, keepdim);
            
            // Test with negative dimension
            int64_t neg_dim = dim - ndim;
            auto result2 = torch::amax(input_tensor, neg_dim, keepdim);
        }

        // Test 2: Multiple dimensions reduction
        if (offset + 1 < Size && ndim > 1) {
            std::vector<int64_t> dims;
            int num_dims = std::min(static_cast<int>(Data[offset] % ndim) + 1, static_cast<int>(ndim));
            offset++;
            
            for (int i = 0; i < num_dims && offset < Size; i++) {
                int64_t dim = static_cast<int64_t>(Data[offset]) % ndim;
                // Avoid duplicate dimensions
                if (std::find(dims.begin(), dims.end(), dim) == dims.end()) {
                    dims.push_back(dim);
                }
                offset++;
            }
            
            if (!dims.empty()) {
                bool keepdim = (offset < Size) ? (Data[offset] % 2 == 0) : false;
                offset++;
                
                auto result3 = torch::amax(input_tensor, dims, keepdim);
            }
        }

        // Test 3: All dimensions (equivalent to global max)
        if (offset < Size) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < ndim; i++) {
                all_dims.push_back(i);
            }
            
            bool keepdim = (Data[offset] % 2 == 0);
            offset++;
            
            auto result4 = torch::amax(input_tensor, all_dims, keepdim);
        }

        // Test 4: Edge cases with different tensor types
        if (offset < Size) {
            // Test with different dtypes if possible
            auto dtype_idx = Data[offset] % 4;
            offset++;
            
            torch::Tensor converted_tensor;
            switch (dtype_idx) {
                case 0:
                    if (input_tensor.dtype() != torch::kFloat32) {
                        converted_tensor = input_tensor.to(torch::kFloat32);
                    } else {
                        converted_tensor = input_tensor;
                    }
                    break;
                case 1:
                    if (input_tensor.dtype() != torch::kFloat64) {
                        converted_tensor = input_tensor.to(torch::kFloat64);
                    } else {
                        converted_tensor = input_tensor;
                    }
                    break;
                case 2:
                    if (input_tensor.dtype() != torch::kInt32) {
                        converted_tensor = input_tensor.to(torch::kInt32);
                    } else {
                        converted_tensor = input_tensor;
                    }
                    break;
                case 3:
                    if (input_tensor.dtype() != torch::kInt64) {
                        converted_tensor = input_tensor.to(torch::kInt64);
                    } else {
                        converted_tensor = input_tensor;
                    }
                    break;
            }
            
            if (converted_tensor.defined()) {
                int64_t dim = 0;
                if (converted_tensor.dim() > 0) {
                    dim = (offset < Size) ? static_cast<int64_t>(Data[offset]) % converted_tensor.dim() : 0;
                }
                auto result5 = torch::amax(converted_tensor, dim);
            }
        }

        // Test 5: Test with output tensor (pre-allocated)
        if (offset < Size && ndim > 0) {
            int64_t dim = static_cast<int64_t>(Data[offset]) % ndim;
            offset++;
            
            bool keepdim = (offset < Size) ? (Data[offset] % 2 == 0) : false;
            offset++;
            
            // Calculate expected output shape
            auto expected_shape = input_tensor.sizes().vec();
            if (keepdim) {
                expected_shape[dim] = 1;
            } else {
                expected_shape.erase(expected_shape.begin() + dim);
            }
            
            if (!expected_shape.empty() || keepdim) {
                auto out_tensor = torch::empty(expected_shape, input_tensor.options());
                torch::amax_out(out_tensor, input_tensor, dim, keepdim);
            }
        }

        // Test 6: Special values (if tensor contains them)
        if (input_tensor.dtype().is_floating_point()) {
            // Create tensor with special values
            auto special_tensor = input_tensor.clone();
            if (special_tensor.numel() > 0) {
                auto flat = special_tensor.flatten();
                if (offset < Size && flat.numel() > 0) {
                    int idx = Data[offset] % flat.numel();
                    offset++;
                    
                    // Inject special values
                    if (offset < Size) {
                        switch (Data[offset] % 4) {
                            case 0: flat[idx] = std::numeric_limits<float>::infinity(); break;
                            case 1: flat[idx] = -std::numeric_limits<float>::infinity(); break;
                            case 2: flat[idx] = std::numeric_limits<float>::quiet_NaN(); break;
                            case 3: flat[idx] = 0.0f; break;
                        }
                        offset++;
                    }
                }
                
                if (special_tensor.dim() > 0) {
                    int64_t dim = (offset < Size) ? static_cast<int64_t>(Data[offset]) % special_tensor.dim() : 0;
                    auto result6 = torch::amax(special_tensor, dim);
                }
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