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
        
        // Need at least 3 tensors for addmm: input, mat1, mat2
        if (Size < 6) // Minimum bytes needed for basic tensor creation
            return 0;
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mat1 tensor
        if (offset >= Size)
            return 0;
        torch::Tensor mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mat2 tensor
        if (offset >= Size)
            return 0;
        torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get scalar values for beta and alpha if there's data left
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
        
        // Try different variants of addmm
        try {
            // Variant 1: Basic addmm
            auto result1 = torch::addmm(input, mat1, mat2);
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 2: With beta and alpha
            auto result2 = torch::addmm(input, mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 3: Out variant
            torch::Tensor out = torch::empty_like(input);
            torch::addmm_out(out, input, mat1, mat2);
        } catch (const std::exception&) {
            // Continue to next variant
        }
        
        try {
            // Variant 4: Out variant with beta and alpha
            torch::Tensor out = torch::empty_like(input);
            torch::addmm_out(out, input, mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
            // Continue
        }
        
        try {
            // Variant 5: Method variant
            auto result5 = input.addmm(mat1, mat2);
        } catch (const std::exception&) {
            // Continue
        }
        
        try {
            // Variant 6: Method variant with beta and alpha
            auto result6 = input.addmm(mat1, mat2, beta, alpha);
        } catch (const std::exception&) {
            // Continue
        }
        
        // Try in-place variant if input has appropriate type
        try {
            // Make a copy to avoid modifying the original input
            torch::Tensor input_copy = input.clone();
            input_copy.addmm_(mat1, mat2);
        } catch (const std::exception&) {
            // Continue
        }
        
        try {
            // In-place with beta and alpha
            torch::Tensor input_copy = input.clone();
            input_copy.addmm_(mat1, mat2, beta, alpha);
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
