#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create a tensor
        if (Size < 16) {
            return 0;
        }

        // Extract tensor configuration parameters
        auto tensor_config = extract_tensor_config(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        // Create input tensor with various shapes and dtypes
        torch::Tensor input_tensor;
        try {
            input_tensor = create_tensor_from_config(tensor_config, Data, Size, offset);
        } catch (...) {
            return 0;
        }

        // Test basic msort functionality
        torch::Tensor result1 = torch::msort(input_tensor);

        // Test msort with different tensor configurations
        if (input_tensor.numel() > 0) {
            // Test with contiguous tensor
            torch::Tensor contiguous_input = input_tensor.contiguous();
            torch::Tensor result2 = torch::msort(contiguous_input);

            // Test with non-contiguous tensor (if possible)
            if (input_tensor.dim() >= 2) {
                torch::Tensor transposed = input_tensor.transpose(0, -1);
                torch::Tensor result3 = torch::msort(transposed);
            }

            // Test with different data types if we have enough data
            if (offset + 1 < Size) {
                uint8_t dtype_choice = Data[offset++];
                
                try {
                    torch::Tensor converted_tensor;
                    switch (dtype_choice % 6) {
                        case 0:
                            converted_tensor = input_tensor.to(torch::kFloat32);
                            break;
                        case 1:
                            converted_tensor = input_tensor.to(torch::kFloat64);
                            break;
                        case 2:
                            converted_tensor = input_tensor.to(torch::kInt32);
                            break;
                        case 3:
                            converted_tensor = input_tensor.to(torch::kInt64);
                            break;
                        case 4:
                            converted_tensor = input_tensor.to(torch::kInt16);
                            break;
                        case 5:
                            converted_tensor = input_tensor.to(torch::kInt8);
                            break;
                    }
                    torch::Tensor result4 = torch::msort(converted_tensor);
                } catch (...) {
                    // Some dtype conversions might fail, continue testing
                }
            }

            // Test with edge case tensors
            if (input_tensor.dim() > 0) {
                // Test with single element tensor
                torch::Tensor single_elem = input_tensor.flatten().slice(0, 0, 1);
                torch::Tensor result5 = torch::msort(single_elem);

                // Test with empty tensor (if possible)
                try {
                    torch::Tensor empty_tensor = torch::empty({0}, input_tensor.options());
                    torch::Tensor result6 = torch::msort(empty_tensor);
                } catch (...) {
                    // Empty tensor might not be supported
                }
            }

            // Test with different tensor shapes
            if (input_tensor.numel() >= 4 && offset + 1 < Size) {
                uint8_t reshape_choice = Data[offset++];
                try {
                    std::vector<int64_t> new_shape;
                    int64_t total_elements = input_tensor.numel();
                    
                    switch (reshape_choice % 4) {
                        case 0: // 1D
                            new_shape = {total_elements};
                            break;
                        case 1: // 2D
                            if (total_elements >= 2) {
                                int64_t dim0 = std::max(1L, total_elements / 2);
                                int64_t dim1 = total_elements / dim0;
                                new_shape = {dim0, dim1};
                            } else {
                                new_shape = {total_elements};
                            }
                            break;
                        case 2: // 3D
                            if (total_elements >= 8) {
                                new_shape = {2, 2, total_elements / 4};
                            } else {
                                new_shape = {total_elements};
                            }
                            break;
                        case 3: // 4D
                            if (total_elements >= 16) {
                                new_shape = {2, 2, 2, total_elements / 8};
                            } else {
                                new_shape = {total_elements};
                            }
                            break;
                    }
                    
                    torch::Tensor reshaped = input_tensor.reshape(new_shape);
                    torch::Tensor result7 = torch::msort(reshaped);
                } catch (...) {
                    // Reshape might fail
                }
            }

            // Test with tensors requiring gradient (if supported)
            if (input_tensor.is_floating_point()) {
                try {
                    torch::Tensor grad_tensor = input_tensor.clone().requires_grad_(true);
                    torch::Tensor result8 = torch::msort(grad_tensor);
                } catch (...) {
                    // Gradient computation might not be supported for msort
                }
            }

            // Test with very large values and edge cases
            if (input_tensor.is_floating_point() && input_tensor.numel() > 0) {
                try {
                    torch::Tensor extreme_tensor = input_tensor.clone();
                    extreme_tensor[0] = std::numeric_limits<float>::infinity();
                    if (extreme_tensor.numel() > 1) {
                        extreme_tensor[1] = -std::numeric_limits<float>::infinity();
                    }
                    if (extreme_tensor.numel() > 2) {
                        extreme_tensor[2] = std::numeric_limits<float>::quiet_NaN();
                    }
                    torch::Tensor result9 = torch::msort(extreme_tensor);
                } catch (...) {
                    // Extreme values might cause issues
                }
            }
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input_tensor.numel() > 0) {
            try {
                torch::Tensor cuda_tensor = input_tensor.to(torch::kCUDA);
                torch::Tensor result_cuda = torch::msort(cuda_tensor);
            } catch (...) {
                // CUDA operations might fail
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