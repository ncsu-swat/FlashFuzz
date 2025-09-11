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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dimensions for transpose
        int64_t dim0 = 0;
        int64_t dim1 = 0;
        
        // Get dimensions to transpose if we have enough data
        if (offset + sizeof(int64_t) * 2 <= Size) {
            std::memcpy(&dim0, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply transpose operation
        torch::Tensor result;
        
        // Try different variants of transpose
        if (input_tensor.dim() > 0) {
            // Variant 1: transpose with specified dimensions
            if (Size % 3 == 0) {
                // Modulo operation to create different test cases
                result = torch::transpose(input_tensor, dim0, dim1);
            }
            // Variant 2: transpose with dimensions clamped to tensor rank
            else if (Size % 3 == 1) {
                int64_t rank = input_tensor.dim();
                if (rank > 1) {
                    // Ensure dimensions are within valid range for the tensor
                    dim0 = std::abs(dim0) % rank;
                    dim1 = std::abs(dim1) % rank;
                    result = torch::transpose(input_tensor, dim0, dim1);
                } else {
                    // For rank 1 or 0, just use the tensor as is
                    result = input_tensor;
                }
            }
            // Variant 3: transpose with potentially negative dimensions
            else {
                // Allow negative dimensions (which are valid in PyTorch)
                result = torch::transpose(input_tensor, dim0, dim1);
            }
        } else {
            // For scalar tensors, transpose is identity
            result = input_tensor;
        }
        
        // Verify the result is a valid tensor
        if (!result.defined()) {
            throw std::runtime_error("Transpose operation returned undefined tensor");
        }
        
        // Optional: perform some operation on the result to ensure it's used
        auto sum = result.sum();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
