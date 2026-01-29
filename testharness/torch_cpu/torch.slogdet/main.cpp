#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

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
        // Skip if we don't have enough data
        if (Size < 4) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Use first byte to determine matrix size (2-8)
        int64_t matrix_size = 2 + (Data[offset++] % 7);
        
        // Use second byte to determine batch dimensions (0-2)
        int batch_dims = Data[offset++] % 3;
        
        // Use third byte to determine dtype
        uint8_t dtype_selector = Data[offset++] % 3;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Build the shape for the tensor
        std::vector<int64_t> shape;
        for (int i = 0; i < batch_dims && offset < Size; i++) {
            int64_t batch_size = 1 + (Data[offset++] % 4);  // batch size 1-4
            shape.push_back(batch_size);
        }
        shape.push_back(matrix_size);
        shape.push_back(matrix_size);
        
        // Calculate total elements needed
        int64_t total_elements = 1;
        for (auto s : shape) {
            total_elements *= s;
        }
        
        // Create tensor with remaining data
        torch::Tensor input;
        if (offset < Size) {
            // Create tensor from fuzzer data
            torch::Tensor raw_data = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            
            // Flatten and resize to match our target shape
            raw_data = raw_data.flatten().to(dtype);
            
            if (raw_data.numel() >= total_elements) {
                input = raw_data.slice(0, 0, total_elements).reshape(shape);
            } else if (raw_data.numel() > 0) {
                // Repeat data to fill the shape
                int64_t repeats = (total_elements + raw_data.numel() - 1) / raw_data.numel();
                auto repeated = raw_data.repeat({repeats});
                input = repeated.slice(0, 0, total_elements).reshape(shape);
            } else {
                // Create random tensor if no valid data
                input = torch::randn(shape, torch::dtype(dtype));
            }
        } else {
            // Create random tensor
            input = torch::randn(shape, torch::dtype(dtype));
        }
        
        // Ensure tensor is contiguous and has correct dtype for slogdet
        input = input.contiguous();
        
        // Inner try-catch for expected failures (singular matrices, etc.)
        try {
            // Apply slogdet operation
            auto result = torch::slogdet(input);
            
            // Unpack the result (sign, logabsdet)
            auto sign = std::get<0>(result);
            auto logabsdet = std::get<1>(result);
            
            // Force computation by accessing values
            if (sign.numel() == 1) {
                volatile auto sign_val = sign.item<double>();
                volatile auto logabsdet_val = logabsdet.item<double>();
                (void)sign_val;
                (void)logabsdet_val;
            } else if (sign.numel() > 0) {
                // For batched input, sum to force computation
                volatile auto sign_sum = sign.sum().item<double>();
                volatile auto logabsdet_sum = logabsdet.sum().item<double>();
                (void)sign_sum;
                (void)logabsdet_sum;
            }
            
            // Additional checks to exercise more code paths
            auto sign_finite = torch::isfinite(sign);
            auto logabsdet_finite = torch::isfinite(logabsdet);
            
        } catch (const c10::Error& e) {
            // Expected failures for singular/ill-conditioned matrices - ignore silently
        } catch (const std::runtime_error& e) {
            // Expected runtime errors - ignore silently
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}