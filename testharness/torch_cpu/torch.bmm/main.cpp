#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Ensure we have at least a few bytes to make decisions
        if (Size < 2)
        {
            return 0;
        }

        // Strategy Selection:
        // Byte 0 determines if we use "Guided" mode (forcing valid ranks/shapes) 
        // or "Wild" mode (random tensors).
        // 0-192: Guided (High probability of valid shapes to hit kernel)
        // 193-255: Wild (Fuzz validation logic with random shapes)
        bool guided_mode = (Data[offset++] < 193);

        torch::Tensor input, mat2;
        c10::optional<torch::Tensor> out = c10::nullopt;

        if (guided_mode)
        {
            // --- Guided Construction for bmm ---
            // bmm requires:
            // input: (b, n, m)
            // mat2:  (b, m, p)
            // Same dtype.

            // 1. Parse Dtype (1 byte)
            if (offset >= Size) return 0;
            uint8_t dtype_byte = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_byte);
            size_t element_size = c10::elementSize(dtype);
            if (element_size == 0) element_size = 1;

            // 2. Parse Dimensions (4 bytes: b, n, m, p)
            if (offset + 4 > Size) return 0;
            
            // Use modulo to keep dimensions reasonable for fuzzing speed and memory
            // b (batch size)
            int64_t b = Data[offset++] % 16; 
            // n, m, p (matrix dims) - ensure at least 1 if possible, but allow 0 for edge cases
            int64_t n = Data[offset++] % 64;
            int64_t m = Data[offset++] % 64;
            int64_t p = Data[offset++] % 64;

            // 3. Create 'input' tensor (b, n, m)
            int64_t input_numel = b * n * m;
            // parseTensorData handles boundary checks on Size vs requested elements
            std::vector<uint8_t> input_buf = fuzzer_utils::parseTensorData(
                Data, offset, Size, input_numel, element_size
            );

            auto options = torch::TensorOptions().dtype(dtype);

            // If we ran out of data, createTensor usually makes an empty tensor or throws.
            // Here we manually handle it to ensure shape is correct even if data is garbage/zeros.
            if (input_buf.size() < static_cast<size_t>(input_numel) * element_size)
            {
                input = torch::empty({b, n, m}, options); // Uninitialized data
            }
            else
            {
                // Clone to own the memory
                input = torch::from_blob(input_buf.data(), {b, n, m}, options).clone();
            }

            // 4. Create 'mat2' tensor (b, m, p)
            int64_t mat2_numel = b * m * p;
            std::vector<uint8_t> mat2_buf = fuzzer_utils::parseTensorData(
                Data, offset, Size, mat2_numel, element_size
            );

            if (mat2_buf.size() < static_cast<size_t>(mat2_numel) * element_size)
            {
                mat2 = torch::empty({b, m, p}, options);
            }
            else
            {
                mat2 = torch::from_blob(mat2_buf.data(), {b, m, p}, options).clone();
            }

            // 5. Optional 'out' tensor
            // Consume 1 byte for decision if available
            if (offset < Size)
            {
                uint8_t out_decision = Data[offset++];
                if (out_decision % 2 == 0)
                {
                    // Construct 'out'
                    // To test valid path: shape should be (b, n, p)
                    // To test invalid path: perturb shape slightly based on decision
                    std::vector<int64_t> out_shape = {b, n, p};
                    
                    if (out_decision > 200) {
                        // Deliberately wrong shape
                        out_shape[2] += 1; 
                    }
                    
                    out = torch::empty(out_shape, options);
                }
            }
        }
        else
        {
            // --- Wild Construction ---
            // Use generic creator. This will often produce Rank != 3 or Dtype mismatch.
            // This is useful for verifying robustness of validation code.
            input = fuzzer_utils::createTensor(Data, Size, offset);
            mat2 = fuzzer_utils::createTensor(Data, Size, offset);

            // Optional out
            if (offset < Size && (Data[offset++] % 2 == 0))
            {
                out = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }

        // --- Execution ---
        if (out.has_value())
        {
            torch::bmm_out(*out, input, mat2);
        }
        else
        {
            torch::bmm(input, mat2);
        }

    }
    catch (const c10::Error &e)
    {
        // Expected exceptions from PyTorch (Shape mismatch, Dtype mismatch, etc.)
        // We silence these as they are valid behavior for invalid inputs.
        return 0;
    }
    catch (const std::runtime_error &e)
    {
        // Expected exceptions from fuzzer_utils (e.g. parsing exhausted data)
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}