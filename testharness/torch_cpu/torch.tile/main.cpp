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
        
        // Need at least a few bytes to create a tensor and reps
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least one byte left for rank
        if (offset >= Size) {
            return 0;
        }
        
        // Parse rank for repetition dimensions (limit to reasonable size)
        uint8_t reps_rank_byte = Data[offset++];
        uint8_t reps_rank = fuzzer_utils::parseRank(reps_rank_byte);
        // Limit reps rank to avoid memory issues
        if (reps_rank > 4) {
            reps_rank = 4;
        }
        
        // Parse repetition dimensions
        std::vector<int64_t> reps;
        if (offset < Size && reps_rank > 0) {
            reps = fuzzer_utils::parseShape(Data, offset, Size, reps_rank);
        }
        
        // Ensure reps are valid (positive and not too large to avoid OOM)
        // If empty, add default
        if (reps.empty()) {
            reps.push_back(2);
        }
        
        // Clamp repetition values to reasonable range [1, 10]
        for (auto& r : reps) {
            if (r <= 0) {
                r = 1;
            } else if (r > 10) {
                r = 10;
            }
        }
        
        // Apply torch.tile operation
        torch::Tensor result = torch::tile(input_tensor, reps);
        
        // Perform some operations on the result to ensure it's used
        auto sum = result.sum();
        
        // Access the sum value (single element)
        if (sum.numel() > 0) {
            auto sum_val = sum.item<float>();
            (void)sum_val;
        }
        
        // Additional coverage: check result shape
        auto result_sizes = result.sizes();
        (void)result_sizes;
        
        // Test contiguous
        auto contiguous_result = result.contiguous();
        (void)contiguous_result;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}