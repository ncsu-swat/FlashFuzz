#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result

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
        
        // Need at least a few bytes for tensor creation and dropout parameters
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dropout probability from the input data
        float p = 0.5; // Default value
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Clamp p to valid range [0, 1]
            p = std::abs(p);
            p = p - std::floor(p); // Get fractional part to ensure 0 <= p <= 1
        }
        
        // Extract train flag from the input data
        bool train = true; // Default value
        if (offset < Size) {
            train = Data[offset++] & 0x1; // Use lowest bit to determine boolean value
        }
        
        // Call native_dropout
        auto result = torch::native_dropout(input, p, train);
        
        // Unpack the result (output tensor and mask tensor)
        torch::Tensor output = std::get<0>(result);
        torch::Tensor mask = std::get<1>(result);
        
        // Verify the output is valid by performing some operations on it
        auto sum = output.sum();
        
        // Verify the mask is valid by performing some operations on it
        auto mask_sum = mask.sum();
        
        // Test edge case: zero probability
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                auto zero_p_result = torch::native_dropout(input, 0.0, train);
                torch::Tensor zero_p_output = std::get<0>(zero_p_result);
                torch::Tensor zero_p_mask = std::get<1>(zero_p_result);
                (void)zero_p_output.sum();
                (void)zero_p_mask.sum();
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test edge case: probability of 1
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                auto one_p_result = torch::native_dropout(input, 1.0, train);
                torch::Tensor one_p_output = std::get<0>(one_p_result);
                torch::Tensor one_p_mask = std::get<1>(one_p_result);
                (void)one_p_output.sum();
                (void)one_p_mask.sum();
            } catch (...) {
                // Silently ignore expected failures
            }
        }
        
        // Test with train=false
        if (offset < Size && (Data[offset++] & 0x1)) {
            try {
                auto no_train_result = torch::native_dropout(input, p, false);
                torch::Tensor no_train_output = std::get<0>(no_train_result);
                torch::Tensor no_train_mask = std::get<1>(no_train_result);
                (void)no_train_output.sum();
                (void)no_train_mask.sum();
            } catch (...) {
                // Silently ignore expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}