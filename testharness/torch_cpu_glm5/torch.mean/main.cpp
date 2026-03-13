#include "fuzzer_utils.h"
#include <iostream>
#include <vector>
#include <c10/util/Optional.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
    try {
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Create input tensor using fuzzer utils
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);

        // Parse operation configuration from remaining bytes
        uint8_t config_byte = (offset < Size) ? Data[offset++] : 0;
        uint8_t dim_config = (offset < Size) ? Data[offset++] : 0;
        uint8_t keepdim_byte = (offset < Size) ? Data[offset++] : 0;
        uint8_t dtype_config = (offset < Size) ? Data[offset++] : 0;

        bool keepdim = (keepdim_byte % 2) == 1;

        // Determine operation mode:
        // 0: torch.mean(input)
        // 1: torch.mean(input, dim)
        // 2: torch.mean(input, dim, keepdim)
        // 3: torch.mean(input, dim_list)
        // 4: torch.mean(input, dtype=...)
        uint8_t op_mode = config_byte % 5;

        // Determine dtype override if needed
        c10::optional<torch::ScalarType> dtype_override;
        if (op_mode == 4 || (dtype_config % 4 == 0)) {
            // Select a valid dtype for mean output
            torch::ScalarType dtypes[] = {
                torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16
            };
            dtype_override = dtypes[dtype_config % 4];
        }

        // Execute the operation based on mode
        torch::Tensor result;

        try {
            switch (op_mode) {
                case 0: {
                    // Simple mean over all elements
                    if (dtype_override.has_value()) {
                        result = torch::mean(input_tensor, *dtype_override);
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                }

                case 1: {
                    // Mean with single dimension
                    int64_t rank = input_tensor.dim();
                    if (rank > 0) {
                        int64_t dim = static_cast<int64_t>(dim_config % rank);
                        // Handle negative dim interpretation
                        if (dim_config % 3 == 2) {
                            dim = dim - rank; // Test negative dim
                        }
                        if (dtype_override.has_value()) {
                            result = torch::mean(input_tensor, {dim}, false, *dtype_override);
                        } else {
                            result = torch::mean(input_tensor, {dim}, false);
                        }
                    } else {
                        // Scalar tensor, just compute mean
                        result = torch::mean(input_tensor);
                    }
                    break;
                }

                case 2: {
                    // Mean with dimension and keepdim
                    int64_t rank = input_tensor.dim();
                    if (rank > 0) {
                        int64_t dim = static_cast<int64_t>(dim_config % rank);
                        if (dtype_override.has_value()) {
                            result = torch::mean(input_tensor, {dim}, keepdim, *dtype_override);
                        } else {
                            result = torch::mean(input_tensor, {dim}, keepdim);
                        }
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                }

                case 3: {
                    // Mean with multiple dimensions
                    int64_t rank = input_tensor.dim();
                    if (rank >= 2) {
                        std::vector<int64_t> dims;
                        int num_dims = 1 + (dim_config % static_cast<int>(rank));
                        for (int i = 0; i < num_dims && static_cast<int64_t>(dims.size()) < rank; ++i) {
                            int64_t d = (dim_config + i * 7) % rank;
                            bool already_added = false;
                            for (auto added_dim : dims) {
                                if (added_dim == d) {
                                    already_added = true;
                                    break;
                                }
                            }
                            if (!already_added) {
                                dims.push_back(d);
                            }
                        }
                        if (dtype_override.has_value()) {
                            result = torch::mean(input_tensor, dims, keepdim, *dtype_override);
                        } else {
                            result = torch::mean(input_tensor, dims, keepdim);
                        }
                    } else if (rank == 1) {
                        result = torch::mean(input_tensor, {0}, keepdim);
                    } else {
                        result = torch::mean(input_tensor);
                    }
                    break;
                }

                case 4: {
                    // Explicit dtype testing
                    torch::ScalarType target_dtype;
                    switch (dtype_config % 4) {
                        case 0: target_dtype = torch::kFloat; break;
                        case 1: target_dtype = torch::kDouble; break;
                        case 2: target_dtype = torch::kHalf; break;
                        default: target_dtype = torch::kBFloat16; break;
                    }
                    result = torch::mean(input_tensor, target_dtype);
                    break;
                }
            }

            // Access result to ensure computation is performed
            if (result.defined() && result.numel() > 0) {
                // Force evaluation by accessing data
                volatile auto ptr = result.data_ptr();
                (void)ptr;
            }

        } catch (const c10::Error &e) {
            // Handle PyTorch-specific errors (e.g., invalid dim, dtype mismatch)
            // These are expected for fuzzing edge cases
            std::cout << "PyTorch error (expected for edge cases): " << e.what_without_backtrace() << std::endl;
        } catch (const std::exception &e) {
            std::cout << "Standard exception during mean operation: " << e.what() << std::endl;
        }

    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}