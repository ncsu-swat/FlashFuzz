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
        
        // Need at least 3 tensors for baddbmm: input, batch1, batch2
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create batch1 tensor
        if (offset >= Size)
            return 0;
        torch::Tensor batch1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create batch2 tensor
        if (offset >= Size)
            return 0;
        torch::Tensor batch2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse beta and alpha values if there's data left
        double beta = 1.0;
        double alpha = 1.0;
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&beta, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply baddbmm operation
        torch::Tensor result;
        
        // Try different variants of baddbmm
        if (offset < Size) {
            uint8_t variant = Data[offset++] % 3;
            
            switch (variant) {
                case 0:
                    // Standard baddbmm
                    result = torch::baddbmm(input, batch1, batch2, beta, alpha);
                    break;
                case 1:
                    // baddbmm with default alpha=1
                    result = torch::baddbmm(input, batch1, batch2, beta);
                    break;
                case 2:
                    // baddbmm with default beta=1, alpha=1
                    result = torch::baddbmm(input, batch1, batch2);
                    break;
            }
        } else {
            // Default to standard baddbmm if no more data
            result = torch::baddbmm(input, batch1, batch2, beta, alpha);
        }
        
        // Perform a simple operation on the result to ensure it's used
        auto sum = result.sum();
        
        // Try in-place version if there's more data
        if (offset < Size && (Data[offset++] % 2 == 0)) {
            // Clone input to avoid modifying the original tensor
            auto input_clone = input.clone();
            input_clone.baddbmm_(batch1, batch2, beta, alpha);
            auto sum2 = input_clone.sum();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
