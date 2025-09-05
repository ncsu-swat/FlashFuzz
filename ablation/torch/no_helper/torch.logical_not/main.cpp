#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 8) {
            return 0;  // Need minimum bytes for basic tensor creation
        }

        size_t offset = 0;
        
        // Extract parameters from fuzzer input
        uint8_t dtype_selector = Data[offset++] % 10;
        uint8_t ndims = (Data[offset++] % 5) + 1;  // 1-5 dimensions
        bool use_out_tensor = Data[offset++] & 1;
        uint8_t out_dtype_selector = Data[offset++] % 10;
        bool make_non_contiguous = Data[offset++] & 1;
        
        // Build shape vector
        std::vector<int64_t> shape;
        size_t total_elements = 1;
        for (size_t i = 0; i < ndims && offset < Size; ++i) {
            int64_t dim = (Data[offset++] % 8);  // 0-7 to allow empty tensors
            shape.push_back(dim);
            total_elements *= dim;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 100000) {
            return 0;
        }
        
        // Create input tensor with various dtypes
        torch::Tensor input;
        torch::TensorOptions options;
        
        switch (dtype_selector) {
            case 0:
                options = torch::TensorOptions().dtype(torch::kBool);
                break;
            case 1:
                options = torch::TensorOptions().dtype(torch::kInt8);
                break;
            case 2:
                options = torch::TensorOptions().dtype(torch::kInt16);
                break;
            case 3:
                options = torch::TensorOptions().dtype(torch::kInt32);
                break;
            case 4:
                options = torch::TensorOptions().dtype(torch::kInt64);
                break;
            case 5:
                options = torch::TensorOptions().dtype(torch::kFloat16);
                break;
            case 6:
                options = torch::TensorOptions().dtype(torch::kFloat32);
                break;
            case 7:
                options = torch::TensorOptions().dtype(torch::kFloat64);
                break;
            case 8:
                options = torch::TensorOptions().dtype(torch::kUInt8);
                break;
            default:
                options = torch::TensorOptions().dtype(torch::kFloat32);
        }
        
        // Create input tensor
        if (total_elements == 0) {
            input = torch::empty(shape, options);
        } else if (offset + total_elements <= Size) {
            // Use fuzzer data for tensor values
            input = torch::empty(shape, options);
            
            if (options.dtype() == torch::kBool) {
                auto accessor = input.flatten().accessor<bool, 1>();
                for (int64_t i = 0; i < total_elements; ++i) {
                    accessor[i] = Data[offset++] & 1;
                }
            } else if (options.dtype() == torch::kInt8) {
                auto accessor = input.flatten().accessor<int8_t, 1>();
                for (int64_t i = 0; i < total_elements; ++i) {
                    accessor[i] = static_cast<int8_t>(Data[offset++]);
                }
            } else if (options.dtype() == torch::kUInt8) {
                auto accessor = input.flatten().accessor<uint8_t, 1>();
                for (int64_t i = 0; i < total_elements; ++i) {
                    accessor[i] = Data[offset++];
                }
            } else {
                // For other types, use random values
                input = torch::randn(shape, options);
                if (offset < Size) {
                    float scale = static_cast<float>(Data[offset++]) / 255.0f * 20.0f - 10.0f;
                    input.mul_(scale);
                }
            }
        } else {
            // Not enough data, use random tensor
            input = torch::randn(shape, options);
        }
        
        // Make non-contiguous if requested
        if (make_non_contiguous && ndims > 1 && shape[0] > 1) {
            input = input.transpose(0, ndims - 1);
        }
        
        // Test logical_not with and without out tensor
        if (use_out_tensor && total_elements > 0) {
            torch::TensorOptions out_options;
            switch (out_dtype_selector) {
                case 0:
                    out_options = torch::TensorOptions().dtype(torch::kBool);
                    break;
                case 1:
                    out_options = torch::TensorOptions().dtype(torch::kInt8);
                    break;
                case 2:
                    out_options = torch::TensorOptions().dtype(torch::kInt16);
                    break;
                case 3:
                    out_options = torch::TensorOptions().dtype(torch::kInt32);
                    break;
                case 4:
                    out_options = torch::TensorOptions().dtype(torch::kFloat32);
                    break;
                case 5:
                    out_options = torch::TensorOptions().dtype(torch::kUInt8);
                    break;
                default:
                    out_options = torch::TensorOptions().dtype(torch::kBool);
            }
            
            torch::Tensor out = torch::empty(shape, out_options);
            torch::logical_not_out(out, input);
            
            // Verify output was written
            if (out.numel() > 0) {
                out.sum();  // Force computation
            }
        } else {
            // Test without out tensor
            torch::Tensor result = torch::logical_not(input);
            
            // Verify result
            if (result.numel() > 0) {
                result.sum();  // Force computation
            }
        }
        
        // Additional edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            switch (edge_case) {
                case 0:
                    // Test with scalar tensor
                    {
                        torch::Tensor scalar = torch::tensor(0.0);
                        torch::logical_not(scalar);
                    }
                    break;
                case 1:
                    // Test with view
                    if (total_elements > 1) {
                        auto view = input.view({-1});
                        torch::logical_not(view);
                    }
                    break;
                case 2:
                    // Test with slice
                    if (shape.size() > 0 && shape[0] > 1) {
                        auto slice = input.narrow(0, 0, 1);
                        torch::logical_not(slice);
                    }
                    break;
                case 3:
                    // Test with special values
                    {
                        torch::Tensor special = torch::tensor({0.0, 1.0, -1.0, 
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()});
                        torch::logical_not(special);
                    }
                    break;
            }
        }
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}