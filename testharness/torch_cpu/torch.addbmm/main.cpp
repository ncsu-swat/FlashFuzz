#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }

        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);

        // Create batch1 tensor (batch of matrices)
        torch::Tensor batch1;
        if (offset < Size) {
            batch1 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple batch tensor
            batch1 = torch::ones({2, 3, 4});
        }

        // Create batch2 tensor (batch of matrices)
        torch::Tensor batch2;
        if (offset < Size) {
            batch2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple batch tensor
            batch2 = torch::ones({2, 4, 5});
        }

        // Create alpha and beta scalars
        float alpha = 1.0f;
        float beta = 1.0f;
        
        if (offset + 8 <= Size) {
            // Extract alpha and beta from the input data
            std::memcpy(&alpha, Data + offset, sizeof(float));
            offset += sizeof(float);
            std::memcpy(&beta, Data + offset, sizeof(float));
            offset += sizeof(float);
        }

        // Try to apply addbmm with different configurations
        try {
            // Standard addbmm: input + beta * input + alpha * (batch1 @ batch2)
            torch::Tensor result = torch::addbmm(input, batch1, batch2, beta, alpha);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
        }

        // Try with default alpha and beta
        try {
            torch::Tensor result = torch::addbmm(input, batch1, batch2);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
        }

        // Try with only beta specified
        try {
            torch::Tensor result = torch::addbmm(input, batch1, batch2, beta);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
        }

        // Try with out version
        try {
            torch::Tensor out = torch::empty_like(input);
            torch::addbmm_out(out, input, batch1, batch2, beta, alpha);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
        }

        // Try with in-place version
        try {
            torch::Tensor inplace = input.clone();
            inplace.addbmm_(batch1, batch2, beta, alpha);
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected for invalid inputs
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}