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
            // Need minimum bytes for basic tensor creation
            return 0;
        }

        size_t offset = 0;

        // Extract parameters from fuzzer input
        uint8_t rank = Data[offset++] % 5;  // Limit rank to 0-4 for memory constraints
        uint8_t dtype_selector = Data[offset++] % 6;
        bool use_out_tensor = Data[offset++] & 1;
        bool make_non_contiguous = Data[offset++] & 1;
        
        // Build shape vector
        std::vector<int64_t> shape;
        for (size_t i = 0; i < rank && offset < Size; i++) {
            int64_t dim = (Data[offset++] % 10) + (i == 0 ? 0 : 1);  // Allow 0 for first dim only
            shape.push_back(dim);
        }

        // Calculate total elements
        int64_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        
        // Limit total elements to prevent OOM
        if (total_elements > 100000) {
            return 0;
        }

        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kFloat16; break;
            case 5: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat32; break;
        }

        // Create input tensor
        torch::Tensor input;
        
        if (total_elements == 0) {
            // Create empty tensor
            input = torch::empty(shape, torch::dtype(dtype));
        } else {
            // Fill tensor with fuzzer data
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                // For floating point types, create varied values including special cases
                std::vector<float> values;
                values.reserve(total_elements);
                
                for (int64_t i = 0; i < total_elements && offset < Size; i++) {
                    if (offset + 3 < Size) {
                        // Use 4 bytes to create float value
                        float val;
                        memcpy(&val, Data + offset, sizeof(float));
                        offset += sizeof(float);
                        
                        // Occasionally inject special values
                        if (i % 17 == 0 && offset < Size) {
                            uint8_t special = Data[offset++] % 5;
                            switch (special) {
                                case 0: val = std::numeric_limits<float>::infinity(); break;
                                case 1: val = -std::numeric_limits<float>::infinity(); break;
                                case 2: val = std::numeric_limits<float>::quiet_NaN(); break;
                                case 3: val = 0.0f; break;
                                case 4: val = -0.0f; break;
                            }
                        }
                        values.push_back(val);
                    } else {
                        // Use remaining bytes
                        float val = offset < Size ? static_cast<float>(Data[offset++]) / 10.0f - 12.5f : 0.0f;
                        values.push_back(val);
                    }
                }
                
                // Pad with zeros if needed
                while (values.size() < static_cast<size_t>(total_elements)) {
                    values.push_back(0.0f);
                }
                
                auto temp = torch::from_blob(values.data(), shape, torch::kFloat32);
                input = temp.clone().to(dtype);
            } else {
                // For integer types
                std::vector<int32_t> values;
                values.reserve(total_elements);
                
                for (int64_t i = 0; i < total_elements && offset < Size; i++) {
                    int32_t val = offset < Size ? static_cast<int32_t>(Data[offset++]) - 128 : 0;
                    values.push_back(val);
                }
                
                // Pad with zeros if needed
                while (values.size() < static_cast<size_t>(total_elements)) {
                    values.push_back(0);
                }
                
                auto temp = torch::from_blob(values.data(), shape, torch::kInt32);
                input = temp.clone().to(dtype);
            }
        }

        // Make tensor non-contiguous if requested
        if (make_non_contiguous && input.numel() > 1 && input.dim() > 0) {
            // Transpose and then slice to create non-contiguous tensor
            if (input.dim() >= 2) {
                input = input.transpose(0, 1);
            } else if (input.size(0) > 1) {
                // Create a strided view
                input = input.slice(0, 0, input.size(0), 2);
            }
        }

        // Test ceil operation
        if (use_out_tensor && offset < Size) {
            // Create output tensor with potentially different properties
            uint8_t out_dtype_selector = Data[offset++] % 3;
            torch::ScalarType out_dtype;
            
            // For ceil, output dtype should be compatible
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                switch (out_dtype_selector) {
                    case 0: out_dtype = torch::kFloat32; break;
                    case 1: out_dtype = torch::kFloat64; break;
                    case 2: out_dtype = dtype; break;  // Same as input
                    default: out_dtype = dtype; break;
                }
            } else {
                out_dtype = dtype;  // For integer types, keep same dtype
            }
            
            torch::Tensor out = torch::empty(input.sizes(), torch::dtype(out_dtype));
            
            // Call ceil with output tensor
            torch::ceil_out(out, input);
            
            // Verify output was written
            if (out.numel() > 0) {
                auto first_elem = out.flatten()[0];
                (void)first_elem;  // Use to avoid optimization
            }
        } else {
            // Call ceil without output tensor
            torch::Tensor result = torch::ceil(input);
            
            // Access result to ensure computation
            if (result.numel() > 0) {
                auto first_elem = result.flatten()[0];
                (void)first_elem;  // Use to avoid optimization
            }
        }

        // Additional edge case: scalar tensor
        if (offset < Size && Data[offset++] & 1) {
            float scalar_val = offset < Size ? static_cast<float>(Data[offset++]) / 10.0f : 0.0f;
            torch::Tensor scalar = torch::tensor(scalar_val);
            torch::Tensor scalar_result = torch::ceil(scalar);
            (void)scalar_result;
        }

        // Test with requires_grad for floating point tensors
        if ((dtype == torch::kFloat32 || dtype == torch::kFloat64) && 
            offset < Size && Data[offset++] & 1) {
            input.requires_grad_(true);
            torch::Tensor grad_result = torch::ceil(input);
            if (grad_result.requires_grad() && grad_result.numel() > 0) {
                // Trigger backward pass
                grad_result.sum().backward();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}