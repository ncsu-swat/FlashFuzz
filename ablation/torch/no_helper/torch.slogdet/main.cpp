#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions
        auto dims = parseDimensions(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (dims.empty()) return 0;

        // Parse dtype
        auto dtype = parseDType(Data, Size, offset);
        
        // Parse device
        auto device = parseDevice(Data, Size, offset);

        // Create input tensor - must be square matrices for slogdet
        torch::Tensor input;
        if (dims.size() == 2) {
            // For 2D, make it square
            int64_t size = std::min(dims[0], dims[1]);
            size = std::max(size, int64_t(1)); // Ensure at least 1x1
            size = std::min(size, int64_t(100)); // Limit size for performance
            input = torch::randn({size, size}, torch::TensorOptions().dtype(dtype).device(device));
        } else if (dims.size() == 3) {
            // For 3D, make last two dimensions square
            int64_t batch = std::max(dims[0], int64_t(1));
            batch = std::min(batch, int64_t(10)); // Limit batch size
            int64_t size = std::min(dims[1], dims[2]);
            size = std::max(size, int64_t(1));
            size = std::min(size, int64_t(50));
            input = torch::randn({batch, size, size}, torch::TensorOptions().dtype(dtype).device(device));
        } else if (dims.size() == 4) {
            // For 4D, make last two dimensions square
            int64_t batch1 = std::max(dims[0], int64_t(1));
            batch1 = std::min(batch1, int64_t(5));
            int64_t batch2 = std::max(dims[1], int64_t(1));
            batch2 = std::min(batch2, int64_t(5));
            int64_t size = std::min(dims[2], dims[3]);
            size = std::max(size, int64_t(1));
            size = std::min(size, int64_t(20));
            input = torch::randn({batch1, batch2, size, size}, torch::TensorOptions().dtype(dtype).device(device));
        }

        // Test edge cases with special values
        if (offset < Size) {
            uint8_t special_case = Data[offset++];
            switch (special_case % 6) {
                case 0:
                    // Identity matrix
                    input = torch::eye(input.size(-1), torch::TensorOptions().dtype(dtype).device(device));
                    if (input.dim() > 2) {
                        auto shape = input.sizes().vec();
                        shape[shape.size()-2] = input.size(-1);
                        shape[shape.size()-1] = input.size(-1);
                        input = torch::eye(input.size(-1), torch::TensorOptions().dtype(dtype).device(device)).expand(shape);
                    }
                    break;
                case 1:
                    // Zero matrix
                    input.zero_();
                    break;
                case 2:
                    // Matrix with very small values
                    input = input * 1e-10;
                    break;
                case 3:
                    // Matrix with very large values
                    input = input * 1e10;
                    break;
                case 4:
                    // Singular matrix (rank deficient)
                    if (input.size(-1) > 1) {
                        input.select(-1, 0).copy_(input.select(-1, 1));
                    }
                    break;
                case 5:
                    // Add some inf/nan values if floating point
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        if (offset < Size && Data[offset++] % 2 == 0) {
                            input.flatten().index_put_({0}, std::numeric_limits<double>::infinity());
                        } else {
                            input.flatten().index_put_({0}, std::numeric_limits<double>::quiet_NaN());
                        }
                    }
                    break;
            }
        }

        // Call torch.slogdet
        auto result = torch::slogdet(input);
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);

        // Verify output shapes
        auto expected_shape = input.sizes().vec();
        expected_shape.pop_back(); // Remove last dimension
        expected_shape.pop_back(); // Remove second to last dimension
        
        if (sign.sizes().vec() != expected_shape) {
            throw std::runtime_error("Sign tensor has incorrect shape");
        }
        if (logabsdet.sizes().vec() != expected_shape) {
            throw std::runtime_error("Logabsdet tensor has incorrect shape");
        }

        // Verify output dtypes
        if (sign.dtype() != input.dtype()) {
            throw std::runtime_error("Sign tensor has incorrect dtype");
        }
        if (logabsdet.dtype() != input.dtype()) {
            throw std::runtime_error("Logabsdet tensor has incorrect dtype");
        }

        // Verify output devices
        if (sign.device() != input.device()) {
            throw std::runtime_error("Sign tensor has incorrect device");
        }
        if (logabsdet.device() != input.device()) {
            throw std::runtime_error("Logabsdet tensor has incorrect device");
        }

        // Test with different input modifications
        if (offset < Size) {
            uint8_t test_case = Data[offset++];
            switch (test_case % 4) {
                case 0:
                    // Test with transposed input
                    if (input.dim() >= 2) {
                        auto transposed = input.transpose(-2, -1);
                        torch::slogdet(transposed);
                    }
                    break;
                case 1:
                    // Test with contiguous input
                    torch::slogdet(input.contiguous());
                    break;
                case 2:
                    // Test with non-contiguous input
                    if (input.size(-1) > 1) {
                        auto sliced = input.index({"...", torch::indexing::Slice(0, torch::indexing::None, 2)});
                        if (sliced.size(-1) == sliced.size(-2)) {
                            torch::slogdet(sliced);
                        }
                    }
                    break;
                case 3:
                    // Test with cloned input
                    torch::slogdet(input.clone());
                    break;
            }
        }

        // Additional edge case: 1x1 matrix
        if (offset < Size && Data[offset++] % 10 == 0) {
            auto small_input = torch::randn({1, 1}, torch::TensorOptions().dtype(dtype).device(device));
            torch::slogdet(small_input);
        }

        // Test with different batch dimensions
        if (input.dim() > 2 && offset < Size && Data[offset++] % 8 == 0) {
            auto squeezed = input.squeeze(0);
            if (squeezed.dim() >= 2 && squeezed.size(-1) == squeezed.size(-2)) {
                torch::slogdet(squeezed);
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