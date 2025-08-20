#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor and reps
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse repetition dimensions
        // Ensure we have at least one byte left for rank
        if (offset >= Size) {
            return 0;
        }
        
        // Parse rank for repetition dimensions
        uint8_t reps_rank_byte = Data[offset++];
        uint8_t reps_rank = fuzzer_utils::parseRank(reps_rank_byte);
        
        // Parse repetition dimensions
        std::vector<int64_t> reps;
        if (offset < Size) {
            reps = fuzzer_utils::parseShape(Data, offset, Size, reps_rank);
        } else {
            // Default repetition if we don't have enough data
            reps.push_back(2);
        }
        
        // Apply torch.tile operation
        torch::Tensor result;
        
        // Try different ways to call tile
        if (offset < Size && Data[offset] % 2 == 0) {
            // Use the reps vector directly
            result = torch::tile(input_tensor, reps);
        } else {
            // Convert reps to individual arguments
            switch (reps.size()) {
                case 0:
                    result = torch::tile(input_tensor, {});
                    break;
                case 1:
                    result = torch::tile(input_tensor, reps[0]);
                    break;
                case 2:
                    result = torch::tile(input_tensor, {reps[0], reps[1]});
                    break;
                case 3:
                    result = torch::tile(input_tensor, {reps[0], reps[1], reps[2]});
                    break;
                case 4:
                    result = torch::tile(input_tensor, {reps[0], reps[1], reps[2], reps[3]});
                    break;
                default:
                    result = torch::tile(input_tensor, reps);
                    break;
            }
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try to access elements if the tensor is not empty
        if (result.numel() > 0) {
            auto item = result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}