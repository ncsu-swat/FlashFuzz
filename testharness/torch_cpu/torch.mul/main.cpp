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
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Decide how to create the second tensor based on remaining data
        torch::Tensor tensor2;
        if (offset < Size) {
            // Create second tensor from remaining data
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Not enough data for second tensor, use scalar multiplication
            // Use first byte as scalar value
            double scalar_value = static_cast<double>(Data[0]) / 255.0;
            tensor2 = torch::scalar_tensor(scalar_value, tensor1.options());
        }
        
        // Try different variants of multiplication
        try {
            // Variant 1: Using torch::mul directly
            torch::Tensor result1 = torch::mul(tensor1, tensor2);
            (void)result1; // Prevent unused variable warning
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: Using operator*
            torch::Tensor result2 = tensor1 * tensor2;
            (void)result2;
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 3: Using in-place multiplication
            torch::Tensor result3 = tensor1.clone();
            result3.mul_(tensor2);
            (void)result3;
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 4: Using out variant
            torch::Tensor result4 = torch::empty_like(tensor1);
            torch::mul_out(result4, tensor1, tensor2);
            (void)result4;
        } catch (const std::exception&) {
            // Continue
        }
        
        // Try broadcasting with different shapes
        if (offset + 2 < Size) {
            try {
                // Create a tensor with different shape for broadcasting
                uint8_t rank_byte = Data[offset++];
                uint8_t rank = fuzzer_utils::parseRank(rank_byte);
                std::vector<int64_t> shape;
                
                // Create a shape that might be broadcastable
                for (uint8_t i = 0; i < rank && offset < Size; i++) {
                    int64_t dim = (Data[offset++] % 5) + 1; // 1-5 for valid dimensions
                    shape.push_back(dim);
                }
                
                if (shape.empty()) {
                    shape.push_back(1); // Ensure at least one dimension
                }
                
                torch::Tensor broadcast_tensor = torch::ones(shape, tensor1.options());
                torch::Tensor broadcast_result = torch::mul(tensor1, broadcast_tensor);
                (void)broadcast_result;
            } catch (const std::exception&) {
                // Continue
            }
        }
        
        // Try scalar multiplication with different scalar types
        if (offset < Size) {
            try {
                // Integer scalar using torch::Scalar
                int64_t int_scalar = static_cast<int64_t>(Data[offset++]);
                torch::Tensor int_result = torch::mul(tensor1, torch::Scalar(int_scalar));
                (void)int_result;
            } catch (const std::exception&) {
                // Continue
            }
            
            try {
                // Float scalar using torch::Scalar
                float float_scalar = static_cast<float>(Data[offset % Size]) / 255.0f;
                torch::Tensor float_result = torch::mul(tensor1, torch::Scalar(float_scalar));
                (void)float_result;
            } catch (const std::exception&) {
                // Continue
            }
            
            try {
                // Double scalar
                double double_scalar = static_cast<double>(Data[offset % Size]) / 127.5 - 1.0;
                torch::Tensor double_result = torch::mul(tensor1, torch::Scalar(double_scalar));
                (void)double_result;
            } catch (const std::exception&) {
                // Continue
            }
        }
        
        // Try tensor.mul() method variant
        try {
            torch::Tensor method_result = tensor1.mul(tensor2);
            (void)method_result;
        } catch (const std::exception&) {
            // Continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}