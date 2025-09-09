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
        if (Size < 8) {
            return 0;
        }

        // Extract tensor configuration parameters
        int ndim = extractInt(Data, Size, offset) % 6 + 1; // 1-6 dimensions
        std::vector<int64_t> shape;
        
        for (int i = 0; i < ndim; i++) {
            int64_t dim_size = std::abs(extractInt(Data, Size, offset)) % 100 + 1; // 1-100 size per dimension
            shape.push_back(dim_size);
        }

        // Extract dtype
        int dtype_idx = extractInt(Data, Size, offset) % 8;
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kUInt8; break;
            case 6: dtype = torch::kBool; break;
            case 7: dtype = torch::kFloat16; break;
            default: dtype = torch::kFloat32; break;
        }

        // Create tensor with the specified shape and dtype
        torch::Tensor tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));

        // Test torch.numel on the created tensor
        int64_t numel_result = tensor.numel();

        // Verify the result makes sense (should equal product of dimensions)
        int64_t expected_numel = 1;
        for (auto dim : shape) {
            expected_numel *= dim;
        }
        
        if (numel_result != expected_numel) {
            std::cerr << "Unexpected numel result: " << numel_result << " vs expected: " << expected_numel << std::endl;
        }

        // Test edge cases if we have enough data
        if (offset < Size - 4) {
            // Test empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            int64_t empty_numel = empty_tensor.numel();
            if (empty_numel != 0) {
                std::cerr << "Empty tensor numel should be 0, got: " << empty_numel << std::endl;
            }

            // Test scalar tensor
            torch::Tensor scalar_tensor = torch::tensor(42.0);
            int64_t scalar_numel = scalar_tensor.numel();
            if (scalar_numel != 1) {
                std::cerr << "Scalar tensor numel should be 1, got: " << scalar_numel << std::endl;
            }

            // Test tensor with zero dimension
            if (extractInt(Data, Size, offset) % 2 == 0) {
                std::vector<int64_t> zero_shape = shape;
                if (!zero_shape.empty()) {
                    zero_shape[0] = 0;
                    torch::Tensor zero_tensor = torch::zeros(zero_shape, torch::TensorOptions().dtype(dtype));
                    int64_t zero_numel = zero_tensor.numel();
                    if (zero_numel != 0) {
                        std::cerr << "Zero-sized tensor numel should be 0, got: " << zero_numel << std::endl;
                    }
                }
            }
        }

        // Test with different tensor creation methods
        if (offset < Size - 4) {
            int creation_method = extractInt(Data, Size, offset) % 4;
            torch::Tensor test_tensor;
            
            switch (creation_method) {
                case 0:
                    test_tensor = torch::ones(shape, torch::TensorOptions().dtype(dtype));
                    break;
                case 1:
                    test_tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype));
                    break;
                case 2:
                    test_tensor = torch::full(shape, 3.14, torch::TensorOptions().dtype(dtype));
                    break;
                case 3:
                    test_tensor = torch::arange(expected_numel, torch::TensorOptions().dtype(dtype)).reshape(shape);
                    break;
            }
            
            int64_t test_numel = test_tensor.numel();
            if (test_numel != expected_numel) {
                std::cerr << "Test tensor numel mismatch: " << test_numel << " vs expected: " << expected_numel << std::endl;
            }
        }

        // Test with reshaped tensors
        if (offset < Size - 4 && expected_numel > 1) {
            // Try to reshape to a different configuration with same total elements
            std::vector<int64_t> new_shape;
            int64_t remaining = expected_numel;
            int new_ndim = extractInt(Data, Size, offset) % 4 + 1;
            
            for (int i = 0; i < new_ndim - 1; i++) {
                int64_t factor = 2; // Simple factorization
                if (remaining % factor == 0) {
                    new_shape.push_back(factor);
                    remaining /= factor;
                } else {
                    new_shape.push_back(1);
                }
            }
            new_shape.push_back(remaining);
            
            torch::Tensor reshaped = tensor.reshape(new_shape);
            int64_t reshaped_numel = reshaped.numel();
            if (reshaped_numel != expected_numel) {
                std::cerr << "Reshaped tensor numel mismatch: " << reshaped_numel << " vs expected: " << expected_numel << std::endl;
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