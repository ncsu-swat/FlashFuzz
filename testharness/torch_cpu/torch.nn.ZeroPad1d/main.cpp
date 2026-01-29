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
        // Need at least a few bytes for padding values and tensor creation
        if (Size < 6) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values from the input data (limit to reasonable range)
        int64_t padding_left = static_cast<int64_t>(Data[offset++] % 32);
        int64_t padding_right = static_cast<int64_t>(Data[offset++] % 32);
        
        // Extract dimensions for the tensor (ZeroPad1d expects 2D or 3D input)
        uint8_t dim_selector = Data[offset++];
        
        // Create appropriate tensor shape for ZeroPad1d
        // ZeroPad1d expects (C, W) for 2D or (N, C, W) for 3D
        std::vector<int64_t> shape;
        if (dim_selector % 2 == 0) {
            // 2D input: (C, W)
            int64_t C = static_cast<int64_t>((Data[offset++] % 15) + 1);
            int64_t W = static_cast<int64_t>((Data[offset++] % 31) + 1);
            shape = {C, W};
        } else {
            // 3D input: (N, C, W)
            int64_t N = static_cast<int64_t>((Data[offset++] % 7) + 1);
            int64_t C = static_cast<int64_t>((Data[offset++] % 15) + 1);
            int64_t W = (offset < Size) ? static_cast<int64_t>((Data[offset++] % 31) + 1) : 8;
            shape = {N, C, W};
        }
        
        // Create input tensor with appropriate shape
        torch::Tensor input_tensor = torch::randn(shape);
        
        // Decide between single value or pair based on remaining byte if available
        std::vector<int64_t> padding;
        if (offset < Size && Data[offset++] % 2 == 0) {
            // Use symmetric padding (single value applied to both sides)
            padding = {padding_left};
        } else {
            // Use asymmetric padding (left, right)
            padding = {padding_left, padding_right};
        }
        
        // Create ZeroPad1d module with padding parameter
        // Use brace initialization to avoid most vexing parse
        torch::nn::ZeroPad1d zero_pad{torch::nn::ZeroPad1dOptions(padding)};
        
        // Apply padding
        torch::Tensor output_tensor;
        try {
            output_tensor = zero_pad->forward(input_tensor);
        } catch (const c10::Error&) {
            // Silently handle invalid shape/padding combinations
            return 0;
        }
        
        // Force computation to ensure any errors are triggered
        output_tensor.sum().item<float>();
        
        // Verify output shape is correct
        int64_t expected_width;
        if (padding.size() == 1) {
            expected_width = shape.back() + 2 * padding[0];
        } else {
            expected_width = shape.back() + padding[0] + padding[1];
        }
        
        // Check the last dimension matches expected padded width
        if (output_tensor.size(-1) != expected_width) {
            std::cerr << "Unexpected output width" << std::endl;
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}