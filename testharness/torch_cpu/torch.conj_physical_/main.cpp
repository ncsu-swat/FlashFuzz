#include "fuzzer_utils.h"
#include <iostream>

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
        if (Size < 4) {
            return 0;
        }

        size_t offset = 0;

        // Use first byte to determine tensor type
        uint8_t type_selector = Data[0] % 4;
        offset = 1;

        torch::Tensor tensor;

        if (type_selector == 0) {
            // Create a complex float tensor
            torch::Tensor real_part = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            size_t offset2 = 0;
            torch::Tensor imag_part = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            // Ensure both tensors are float type for complex creation
            real_part = real_part.to(torch::kFloat32);
            imag_part = imag_part.to(torch::kFloat32);
            
            // Match shapes
            auto target_sizes = real_part.sizes().vec();
            imag_part = imag_part.expand(target_sizes).clone();
            
            try {
                tensor = torch::complex(real_part, imag_part);
            } catch (...) {
                // Fall back to simple tensor if complex creation fails
                tensor = real_part;
            }
        } else if (type_selector == 1) {
            // Create a complex double tensor
            torch::Tensor real_part = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            size_t offset2 = 0;
            torch::Tensor imag_part = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            
            real_part = real_part.to(torch::kFloat64);
            imag_part = imag_part.to(torch::kFloat64);
            
            auto target_sizes = real_part.sizes().vec();
            imag_part = imag_part.expand(target_sizes).clone();
            
            try {
                tensor = torch::complex(real_part, imag_part);
            } catch (...) {
                tensor = real_part;
            }
        } else {
            // Create a regular (non-complex) tensor
            tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
        }

        // Apply conj_physical_ operation (in-place conjugation)
        tensor.conj_physical_();

        // Access result to ensure operation completed
        volatile auto numel = tensor.numel();
        (void)numel;

        // Test on a contiguous tensor
        if (Size > offset + 2) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
            tensor2 = tensor2.contiguous();
            tensor2.conj_physical_();
        }

        // Test on a non-contiguous tensor (transposed)
        if (Size > offset + 2) {
            size_t new_offset = 0;
            torch::Tensor tensor3 = fuzzer_utils::createTensor(Data + offset, Size - offset, new_offset);
            if (tensor3.dim() >= 2) {
                tensor3 = tensor3.transpose(0, 1);
                try {
                    tensor3.conj_physical_();
                } catch (...) {
                    // Some non-contiguous tensors may not support in-place ops
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}