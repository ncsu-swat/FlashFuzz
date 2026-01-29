#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need enough data for control bytes and tensor values
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract control bytes for configuration
        uint8_t variant = Data[offset++] % 4;
        uint8_t dtype_selector = Data[offset++] % 3;
        uint8_t shape_variant = Data[offset++] % 4;
        int8_t dim_byte = static_cast<int8_t>(Data[offset++]);

        // Select dtype
        torch::Dtype dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }

        auto options = torch::TensorOptions().dtype(dtype);

        // Cross product requires size 3 in the specified dimension
        // Create tensors with valid shapes for cross product
        std::vector<int64_t> shape;
        switch (shape_variant) {
            case 0:
                // Simple 1D vectors of size 3
                shape = {3};
                break;
            case 1:
                // 2D with last dim = 3
                shape = {2, 3};
                break;
            case 2:
                // 2D with first dim = 3
                shape = {3, 2};
                break;
            case 3:
                // 3D with middle dim = 3
                shape = {2, 3, 4};
                break;
        }

        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }

        // Create tensors from fuzzer data
        torch::Tensor input1, input2;

        if (offset + total_elements * 4 * 2 <= Size) {
            // Use fuzzer data to populate tensors
            std::vector<float> data1(total_elements);
            std::vector<float> data2(total_elements);

            for (int64_t i = 0; i < total_elements && offset + 4 <= Size; i++) {
                float val;
                std::memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                // Sanitize to avoid NaN/Inf issues
                if (!std::isfinite(val)) val = 0.0f;
                data1[i] = val;
            }

            for (int64_t i = 0; i < total_elements && offset + 4 <= Size; i++) {
                float val;
                std::memcpy(&val, Data + offset, sizeof(float));
                offset += sizeof(float);
                if (!std::isfinite(val)) val = 0.0f;
                data2[i] = val;
            }

            input1 = torch::from_blob(data1.data(), shape, torch::kFloat32).clone().to(dtype);
            input2 = torch::from_blob(data2.data(), shape, torch::kFloat32).clone().to(dtype);
        } else {
            // Use random tensors if not enough data
            input1 = torch::randn(shape, options);
            input2 = torch::randn(shape, options);
        }

        // Determine dimension for cross product (must have size 3)
        int64_t dim = -1;
        for (int64_t d = 0; d < static_cast<int64_t>(shape.size()); d++) {
            if (shape[d] == 3) {
                dim = d;
                break;
            }
        }

        // Execute different variants based on control byte
        switch (variant) {
            case 0: {
                // Basic cross product with explicit dimension
                try {
                    torch::Tensor result = torch::cross(input1, input2, dim);
                    // Access result to ensure computation
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                } catch (const std::exception&) {
                    // Shape/type mismatch - silently continue
                }
                break;
            }
            case 1: {
                // Cross product with dimension from fuzzer input
                int64_t fuzz_dim = dim_byte % (input1.dim() + 1);
                if (fuzz_dim < 0) fuzz_dim = -fuzz_dim;
                fuzz_dim = fuzz_dim % input1.dim();
                
                try {
                    torch::Tensor result = torch::cross(input1, input2, fuzz_dim);
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                } catch (const std::exception&) {
                    // Expected failure for invalid dimension
                }
                break;
            }
            case 2: {
                // Cross product with contiguous tensors
                try {
                    torch::Tensor cont1 = input1.contiguous();
                    torch::Tensor cont2 = input2.contiguous();
                    torch::Tensor result = torch::cross(cont1, cont2, dim);
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                } catch (const std::exception&) {
                    // Silently continue
                }
                break;
            }
            case 3: {
                // Cross product with transposed/permuted tensors
                if (input1.dim() >= 2 && input2.dim() >= 2) {
                    try {
                        torch::Tensor t1 = input1.transpose(0, input1.dim() - 1).contiguous();
                        torch::Tensor t2 = input2.transpose(0, input2.dim() - 1).contiguous();
                        
                        // Find new dimension with size 3
                        int64_t new_dim = -1;
                        for (int64_t d = 0; d < t1.dim(); d++) {
                            if (t1.size(d) == 3) {
                                new_dim = d;
                                break;
                            }
                        }
                        
                        if (new_dim >= 0) {
                            torch::Tensor result = torch::cross(t1, t2, new_dim);
                            volatile float sum = result.sum().item<float>();
                            (void)sum;
                        }
                    } catch (const std::exception&) {
                        // Silently continue
                    }
                }
                break;
            }
        }

        // Additional test: cross product with out parameter
        try {
            torch::Tensor out = torch::empty_like(input1);
            torch::cross_out(out, input1, input2, dim);
            volatile float sum = out.sum().item<float>();
            (void)sum;
        } catch (const std::exception&) {
            // Silently continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}