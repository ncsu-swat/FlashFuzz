#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse parameters for as_strided_copy
        // Need at least 1 byte for rank of size
        if (offset + 1 >= Size) {
            return 0;
        }
        
        // Parse rank for size
        uint8_t size_rank_byte = Data[offset++];
        uint8_t size_rank = fuzzer_utils::parseRank(size_rank_byte);
        
        // Parse size vector
        std::vector<int64_t> size = fuzzer_utils::parseShape(Data, offset, Size, size_rank);
        
        // Parse stride vector
        if (offset + 1 >= Size) {
            return 0;
        }
        
        uint8_t stride_rank_byte = Data[offset++];
        uint8_t stride_rank = fuzzer_utils::parseRank(stride_rank_byte);
        
        std::vector<int64_t> stride = fuzzer_utils::parseShape(Data, offset, Size, stride_rank);
        
        // Parse storage_offset
        int64_t storage_offset = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&storage_offset, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Make sure size and stride have the same rank
        if (size.size() != stride.size() && !size.empty() && !stride.empty()) {
            // If they don't match, resize the smaller one to match the larger one
            if (size.size() < stride.size()) {
                while (size.size() < stride.size()) {
                    size.push_back(1);
                }
            } else {
                while (stride.size() < size.size()) {
                    stride.push_back(1);
                }
            }
        }
        
        // Apply as_strided_copy operation
        try {
            torch::Tensor result = torch::as_strided_copy(input_tensor, size, stride, storage_offset);
            
            // Perform some operations on the result to ensure it's used
            if (result.defined()) {
                auto sum = result.sum();
                auto mean = result.mean();
                auto std_dev = result.std();
            }
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and handled
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}