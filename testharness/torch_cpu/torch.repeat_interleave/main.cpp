#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operations
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for repeat_interleave
        // We need at least 1 byte for each parameter
        if (offset + 2 > Size) {
            return 0;
        }
        
        // Get repeats parameter - can be a tensor or an int
        bool use_tensor_repeats = Data[offset++] % 2 == 0;
        
        torch::Tensor repeats_tensor;
        int64_t repeats_int = 1;
        
        if (use_tensor_repeats) {
            // Create a tensor for repeats if there's enough data
            if (offset < Size) {
                repeats_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Default to a simple repeats tensor if not enough data
                repeats_tensor = torch::tensor({1, 2, 1}, torch::kInt64);
            }
        } else {
            // Use an integer for repeats
            if (offset < Size) {
                // Get a value between 0 and 10 to avoid excessive memory usage
                repeats_int = static_cast<int64_t>(Data[offset++] % 11);
            }
        }
        
        // Get optional dim parameter
        int64_t dim = -1;
        bool use_dim = false;
        
        if (offset < Size) {
            use_dim = Data[offset++] % 2 == 0;
            
            if (use_dim && offset < Size) {
                // Get a dimension value that could be valid or invalid
                // We don't restrict to valid dimensions to test error handling
                dim = static_cast<int8_t>(Data[offset++]);
            }
        }
        
        // Get optional output_size parameter
        int64_t output_size = -1;
        bool use_output_size = false;
        
        if (offset < Size) {
            use_output_size = Data[offset++] % 2 == 0;
            
            if (use_output_size && offset < Size) {
                // Get an output_size value
                output_size = static_cast<int64_t>(Data[offset++] % 100);
            }
        }
        
        // Apply repeat_interleave with different parameter combinations
        torch::Tensor result;
        
        if (use_tensor_repeats) {
            if (use_dim) {
                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor, dim, output_size);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_tensor, dim);
                }
            } else {
                result = torch::repeat_interleave(input_tensor, repeats_tensor);
            }
        } else {
            if (use_dim) {
                if (use_output_size) {
                    result = torch::repeat_interleave(input_tensor, repeats_int, dim, output_size);
                } else {
                    result = torch::repeat_interleave(input_tensor, repeats_int, dim);
                }
            } else {
                result = torch::repeat_interleave(input_tensor, repeats_int);
            }
        }
        
        // Perform a simple operation on the result to ensure it's used
        auto sum = result.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
