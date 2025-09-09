#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for basic parameters
        if (Size < 16) {
            return 0;
        }

        // Extract input tensor data
        auto tensor_info = extract_tensor_info(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        // Create input tensor with various dtypes that make sense for quantization
        torch::Tensor input;
        switch (tensor_info.dtype_idx % 3) {
            case 0:
                input = create_tensor<float>(tensor_info, Data, Size, offset);
                break;
            case 1:
                input = create_tensor<double>(tensor_info, Data, Size, offset);
                break;
            case 2:
                input = create_tensor<torch::Half>(tensor_info, Data, Size, offset);
                break;
        }

        if (offset >= Size) {
            return 0;
        }

        // Extract numel parameter (number of elements to consider)
        int64_t numel = extract_int64(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        // Clamp numel to reasonable range
        numel = std::max(1L, std::min(numel, input.numel()));

        // Extract reduce_range parameter
        bool reduce_range = extract_bool(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        // Test torch::choose_qparams_optimized with different configurations
        
        // Test 1: Basic call with extracted parameters
        auto result1 = torch::choose_qparams_optimized(input, numel, reduce_range);
        
        // Verify the result is a tuple with scale and zero_point
        auto scale1 = std::get<0>(result1);
        auto zero_point1 = std::get<1>(result1);
        
        // Basic sanity checks
        if (scale1.item<double>() <= 0) {
            std::cout << "Invalid scale: " << scale1.item<double>() << std::endl;
        }
        
        int64_t zp_val = zero_point1.item<int64_t>();
        if (zp_val < 0 || zp_val > 255) {
            std::cout << "Zero point out of range: " << zp_val << std::endl;
        }

        // Test 2: Edge case with numel = 1
        auto result2 = torch::choose_qparams_optimized(input, 1, reduce_range);
        
        // Test 3: Edge case with full tensor
        auto result3 = torch::choose_qparams_optimized(input, input.numel(), reduce_range);
        
        // Test 4: Toggle reduce_range
        auto result4 = torch::choose_qparams_optimized(input, numel, !reduce_range);

        // Test with different tensor shapes and values
        if (offset < Size) {
            // Create a tensor with extreme values
            torch::Tensor extreme_tensor;
            uint8_t extreme_type = Data[offset++] % 4;
            
            switch (extreme_type) {
                case 0: {
                    // All zeros
                    extreme_tensor = torch::zeros_like(input);
                    break;
                }
                case 1: {
                    // All ones
                    extreme_tensor = torch::ones_like(input);
                    break;
                }
                case 2: {
                    // Large positive values
                    extreme_tensor = torch::full_like(input, 1000.0);
                    break;
                }
                case 3: {
                    // Large negative values
                    extreme_tensor = torch::full_like(input, -1000.0);
                    break;
                }
            }
            
            auto result_extreme = torch::choose_qparams_optimized(extreme_tensor, 
                                                                std::min(numel, extreme_tensor.numel()), 
                                                                reduce_range);
        }

        // Test with different tensor devices if CUDA is available
        if (torch::cuda::is_available() && input.numel() > 0) {
            try {
                auto cuda_input = input.to(torch::kCUDA);
                auto result_cuda = torch::choose_qparams_optimized(cuda_input, 
                                                                 std::min(numel, cuda_input.numel()), 
                                                                 reduce_range);
            } catch (const std::exception& e) {
                // CUDA operations might fail, that's okay
            }
        }

        // Test with contiguous and non-contiguous tensors
        if (input.dim() > 1) {
            try {
                auto transposed = input.transpose(0, -1);
                if (!transposed.is_contiguous()) {
                    auto result_noncontig = torch::choose_qparams_optimized(transposed, 
                                                                          std::min(numel, transposed.numel()), 
                                                                          reduce_range);
                }
            } catch (const std::exception& e) {
                // Non-contiguous operations might have restrictions
            }
        }

        // Test with very small numel values
        for (int64_t small_numel : {1, 2, 3}) {
            if (small_numel <= input.numel()) {
                auto result_small = torch::choose_qparams_optimized(input, small_numel, reduce_range);
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