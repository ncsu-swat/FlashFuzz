#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Generate input tensor with various shapes and dtypes
        auto input_tensor = generateTensor(Data, Size, offset);
        if (input_tensor.numel() == 0) {
            return 0; // Skip empty tensors
        }

        // Get tensor dimensions
        int64_t ndim = input_tensor.dim();
        if (ndim == 0) {
            // For scalar tensors, test without dim parameter
            auto result = torch::amin(input_tensor);
            return 0;
        }

        // Test case 1: Single dimension reduction
        if (offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset]) % ndim;
            offset++;
            
            bool keepdim = false;
            if (offset < Size) {
                keepdim = (Data[offset] % 2) == 1;
                offset++;
            }

            auto result1 = torch::amin(input_tensor, dim, keepdim);
            
            // Test with negative dimension
            int64_t neg_dim = dim - ndim;
            auto result1_neg = torch::amin(input_tensor, neg_dim, keepdim);
        }

        // Test case 2: Multiple dimensions reduction
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
                bool keepdim = false;
                if (offset < Size) {
                    keepdim = (Data[offset] % 2) == 1;
                    offset++;
                }
                
                auto result2 = torch::amin(input_tensor, dims, keepdim);
            }
        }

        // Test case 3: All dimensions (equivalent to global min)
        if (ndim > 0) {
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < ndim; i++) {
                all_dims.push_back(i);
            }
            auto result3 = torch::amin(input_tensor, all_dims, false);
            auto result3_keepdim = torch::amin(input_tensor, all_dims, true);
        }

        // Test case 4: Edge cases with special values
        if (input_tensor.dtype() == torch::kFloat32 || input_tensor.dtype() == torch::kFloat64) {
            // Create tensor with special float values
            auto special_tensor = input_tensor.clone();
            if (special_tensor.numel() > 0) {
                auto flat = special_tensor.flatten();
                if (offset < Size && flat.numel() > 0) {
                    int idx = Data[offset] % flat.numel();
                    if (offset + 1 < Size) {
                        uint8_t special_val = Data[offset + 1] % 4;
                        if (special_val == 0) {
                            flat[idx] = std::numeric_limits<float>::infinity();
                        } else if (special_val == 1) {
                            flat[idx] = -std::numeric_limits<float>::infinity();
                        } else if (special_val == 2) {
                            flat[idx] = std::numeric_limits<float>::quiet_NaN();
                        }
                        // special_val == 3: keep original value
                    }
                    offset += 2;
                }
                
                if (ndim > 0) {
                    int64_t dim = 0;
                    auto result_special = torch::amin(special_tensor, dim);
                }
            }
        }

        // Test case 5: Different tensor layouts and memory formats
        if (input_tensor.dim() >= 2) {
            auto transposed = input_tensor.transpose(0, 1);
            auto result_transposed = torch::amin(transposed, 0);
            
            if (input_tensor.dim() == 4) {
                // Test with channels_last format if applicable
                try {
                    auto channels_last = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                    auto result_cl = torch::amin(channels_last, 1);
                } catch (...) {
                    // Ignore if channels_last conversion fails
                }
            }
        }

        // Test case 6: Very large and very small dimensions
        if (ndim > 0) {
            // Test with the last dimension
            auto result_last = torch::amin(input_tensor, ndim - 1);
            
            // Test with dimension 0
            auto result_first = torch::amin(input_tensor, 0);
        }

        // Test case 7: Output tensor parameter
        if (ndim > 0 && offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset]) % ndim;
            
            // Create output tensor with correct shape
            auto input_shape = input_tensor.sizes().vec();
            input_shape[dim] = 1;  // keepdim=true case
            auto out_tensor = torch::empty(input_shape, input_tensor.options());
            
            torch::amin_out(out_tensor, input_tensor, dim, true);
            
            // Test without keepdim
            input_shape.erase(input_shape.begin() + dim);
            if (!input_shape.empty()) {
                auto out_tensor2 = torch::empty(input_shape, input_tensor.options());
                torch::amin_out(out_tensor2, input_tensor, dim, false);
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