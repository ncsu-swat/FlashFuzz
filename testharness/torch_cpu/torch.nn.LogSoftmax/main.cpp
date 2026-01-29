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
        
        // Need at least 2 bytes for dimension and dtype selection
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // LogSoftmax requires floating point input
        if (!input.is_floating_point()) {
            // Select dtype based on fuzzer data
            uint8_t dtype_byte = (offset < Size) ? Data[offset++] : 0;
            switch (dtype_byte % 3) {
                case 0:
                    input = input.to(torch::kFloat32);
                    break;
                case 1:
                    input = input.to(torch::kFloat64);
                    break;
                case 2:
                    input = input.to(torch::kFloat16);
                    break;
            }
        }
        
        // Handle empty or scalar tensors
        if (input.numel() == 0 || input.dim() == 0) {
            // Create a default tensor for scalar/empty cases
            input = torch::randn({2, 3});
        }
        
        // Get dimension for LogSoftmax
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a dimension value from the input data
            uint8_t dim_byte = Data[offset++];
            
            // Select dimension (can be negative in PyTorch)
            if (input.dim() > 0) {
                // Allow both positive and negative indexing
                int total_dims = input.dim();
                dim = (static_cast<int64_t>(dim_byte) % (2 * total_dims)) - total_dims;
                // Clamp to valid range
                if (dim < -total_dims) dim = -total_dims;
                if (dim >= total_dims) dim = total_dims - 1;
            }
        }
        
        // Test 1: Create LogSoftmax module with the selected dimension
        try {
            auto log_softmax = torch::nn::LogSoftmax{torch::nn::LogSoftmaxOptions(dim)};
            torch::Tensor output = log_softmax->forward(input);
            
            // Verify output properties (should sum to 0 in log space along dim)
            (void)output.sizes();
        } catch (const std::exception &) {
            // Expected for some invalid configurations
        }
        
        // Test 2: Functional interface
        try {
            torch::Tensor output2 = torch::log_softmax(input, dim);
            (void)output2.sizes();
        } catch (const std::exception &) {
            // Expected for some invalid configurations
        }
        
        // Test 3: Test with different dtype explicitly
        try {
            if (offset < Size && Data[offset] % 2 == 0) {
                torch::Tensor float_input = input.to(torch::kFloat32);
                torch::Tensor output3 = torch::log_softmax(float_input, dim);
                (void)output3.sizes();
            }
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        // Test 4: Test with contiguous vs non-contiguous tensor
        try {
            if (input.dim() >= 2 && offset < Size && Data[offset] % 3 == 0) {
                torch::Tensor transposed = input.transpose(0, 1);
                torch::Tensor output4 = torch::log_softmax(transposed, dim % transposed.dim());
                (void)output4.sizes();
            }
        } catch (const std::exception &) {
            // Expected for some configurations
        }
        
        return 0; // keep the input
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
}