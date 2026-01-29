#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 8) {
            return 0;
        }
        
        // Extract dimensions for 2D weight matrix
        // fbgemm_linear_quantize_weight expects a 2D float tensor
        uint8_t dim0_byte = (offset < Size) ? Data[offset++] : 1;
        uint8_t dim1_byte = (offset < Size) ? Data[offset++] : 1;
        
        // Create reasonable dimensions (at least 1x1, avoid too large)
        int64_t out_features = (dim0_byte % 64) + 1;  // 1-64
        int64_t in_features = (dim1_byte % 64) + 1;   // 1-64
        
        // Create a 2D float weight tensor (required by fbgemm)
        torch::Tensor weight = torch::randn({out_features, in_features}, 
                                            torch::dtype(torch::kFloat32));
        
        // If we have more data, use it to modify the weight values
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = std::min(remaining / sizeof(float), 
                                          static_cast<size_t>(weight.numel()));
            if (num_elements > 0) {
                auto accessor = weight.accessor<float, 2>();
                for (size_t i = 0; i < num_elements && offset + sizeof(float) <= Size; i++) {
                    float val;
                    memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Sanitize value to avoid NaN/Inf issues
                    if (std::isnan(val) || std::isinf(val)) {
                        val = 0.0f;
                    }
                    int64_t row = i / in_features;
                    int64_t col = i % in_features;
                    if (row < out_features && col < in_features) {
                        accessor[row][col] = val;
                    }
                }
            }
        }
        
        // Call fbgemm_linear_quantize_weight
        // Returns tuple of (quantized_weight, col_offsets, scale)
        auto result = torch::fbgemm_linear_quantize_weight(weight);
        
        // Access the result to ensure it's computed
        // Result is a tuple<Tensor, Tensor, double>
        auto quantized_weight = std::get<0>(result);
        auto col_offsets = std::get<1>(result);
        auto scale = std::get<2>(result);
        
        // Force evaluation to catch any errors
        // Use sum() for tensors, not item() since they're not scalars
        (void)quantized_weight.sum().item<float>();
        (void)col_offsets.sum().item<int>();
        (void)scale;  // scale is a double, not a tensor
        
        // Try with different weight configurations
        if (offset < Size) {
            try {
                // Try with a different shape
                uint8_t dim0_byte2 = (offset < Size) ? Data[offset++] : 1;
                uint8_t dim1_byte2 = (offset < Size) ? Data[offset++] : 1;
                int64_t out_features2 = (dim0_byte2 % 32) + 1;
                int64_t in_features2 = (dim1_byte2 % 32) + 1;
                
                torch::Tensor weight2 = torch::randn({out_features2, in_features2},
                                                     torch::dtype(torch::kFloat32));
                auto result2 = torch::fbgemm_linear_quantize_weight(weight2);
                (void)std::get<0>(result2).numel();
            } catch (...) {
                // Expected failures, ignore silently
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}