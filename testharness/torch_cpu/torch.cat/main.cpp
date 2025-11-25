#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        // Require at least a few bytes for 'dim' and 'num_tensors'
        if (Size < 2)
        {
            return 0;
        }

        size_t offset = 0;

        // 1. Parse 'dim' argument
        // torch.cat accepts an integer dimension. 
        // We interpret the first byte as a signed 8-bit integer. 
        // This efficiently covers common cases: 0, 1, 2 (small ranks) and -1, -2 (negative indexing).
        int64_t dim = static_cast<int64_t>(static_cast<int8_t>(Data[offset++]));

        // 2. Parse number of tensors
        // We determine how many tensors to pass to torch.cat.
        // We limit this to a small range (0-7) to ensure reasonable performance and memory usage.
        // - 0 tensors: Tests handling of empty sequences (should throw).
        // - 1 tensor: Tests identity operation.
        // - 2+ tensors: Tests actual concatenation.
        uint8_t num_tensors_byte = Data[offset++];
        size_t num_tensors = num_tensors_byte % 8; 

        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);

        for (size_t i = 0; i < num_tensors; ++i)
        {
            // fuzzer_utils::createTensor consumes data from the buffer.
            // It throws std::runtime_error if there is insufficient data or invalid metadata.
            tensors.push_back(fuzzer_utils::createTensor(Data, Size, offset));
        }

        // 3. Invoke torch::cat
        // The API signature is cat(TensorList tensors, int64_t dim)
        // std::vector implicitly converts to ArrayRef/TensorList.
        auto result = torch::cat(tensors, dim);

        // We do not need to use 'result'. The goal is to trigger assertions or crashes 
        // within the PyTorch C++ backend.
    }
    catch (const c10::Error &e)
    {
        // Handle PyTorch-specific errors (e.g., "Sizes of tensors must match").
        // These are valid execution paths given random inputs.
        return 0;
    }
    catch (const std::runtime_error &e)
    {
        // Handle parsing errors from fuzzer_utils (e.g., "Input data too small").
        // These imply the input data didn't structure into a valid fuzz case.
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}