#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::max

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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimension parameter from the remaining data
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Ensure dim is within valid range for the tensor
        if (input.dim() > 0) {
            dim = dim % input.dim();
            if (dim < 0) {
                dim += input.dim();
            }
        } else {
            dim = 0;
        }
        
        // Create LogSoftmax module with the dimension parameter using brace initialization
        // to avoid the "most vexing parse" issue
        torch::nn::LogSoftmax logsoftmax{torch::nn::LogSoftmaxOptions(dim)};
        
        // Apply LogSoftmax to the input tensor
        torch::Tensor output = logsoftmax(input);
        
        // Try different ways of calling LogSoftmax
        // 1. Using the functional interface
        torch::Tensor output2 = torch::log_softmax(input, dim);
        
        // 2. Try with different dimension
        int64_t alt_dim = (dim + 1) % std::max(static_cast<int64_t>(1), input.dim());
        torch::nn::LogSoftmax logsoftmax_alt{torch::nn::LogSoftmaxOptions(alt_dim)};
        torch::Tensor output4 = logsoftmax_alt(input);
        
        // 3. Try with a different data type if possible
        if (input.dtype() != torch::kFloat) {
            try {
                torch::Tensor float_input = input.to(torch::kFloat);
                torch::Tensor output5 = logsoftmax(float_input);
            } catch (...) {
                // Silently ignore exceptions from type conversion
            }
        }
        
        // 4. Try with a negative dimension value
        if (input.dim() > 0) {
            int64_t neg_dim = -1;
            torch::nn::LogSoftmax logsoftmax_neg{torch::nn::LogSoftmaxOptions(neg_dim)};
            torch::Tensor output6 = logsoftmax_neg(input);
        }
        
        // 5. Try with double precision
        try {
            torch::Tensor double_input = input.to(torch::kDouble);
            torch::nn::LogSoftmax logsoftmax_double{torch::nn::LogSoftmaxOptions(dim)};
            torch::Tensor output7 = logsoftmax_double(double_input);
        } catch (...) {
            // Silently ignore exceptions
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}