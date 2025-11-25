#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

// --- Helper Function ---
// Manually constructs a tensor with a specific shape and dtype from the fuzz data.
// This ensures we can enforce the matrix multiplication constraints (n x m) * (m x p).
torch::Tensor consume_tensor(const uint8_t* data, size_t& offset, size_t total_size, 
                             const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    
    int64_t num_elements = 1;
    for (auto d : shape) {
        num_elements *= d;
    }
    
    // Handle potential 0-sized element calculation safe-guard
    if (num_elements < 0) num_elements = 0;

    size_t element_size = torch::elementSize(dtype);
    size_t bytes_needed = num_elements * element_size;
    
    // Prepare buffer, zero-initialized
    // We use a vector to manage the raw bytes temporarily
    std::vector<uint8_t> buffer(bytes_needed, 0);
    
    size_t available = (offset < total_size) ? (total_size - offset) : 0;
    size_t to_copy = std::min(available, bytes_needed);
    
    if (to_copy > 0) {
        std::memcpy(buffer.data(), data + offset, to_copy);
        offset += to_copy;
    }
    
    auto options = torch::TensorOptions().dtype(dtype);
    // Create tensor from blob (borrows memory), then clone (owns memory)
    // We must clone because 'buffer' will be destroyed at the end of this function.
    // from_blob is unsafe if the data pointer becomes invalid.
    return torch::from_blob(buffer.data(), shape, options).clone();
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Require enough bytes for configuration: n, m, p, dtype_byte, mode_byte
    if (Size < 5) {
        return 0;
    }

    try
    {
        size_t offset = 0;

        // 1. Parse Dimensions
        // Restrict sizes to [0, 128] to ensure fast execution and reasonable memory usage.
        // Dimensions n, m, p define the matrix shapes: (n x m) and (m x p).
        int64_t n = Data[offset++] % 129;
        int64_t m = Data[offset++] % 129;
        int64_t p = Data[offset++] % 129;

        // 2. Parse Data Type
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);

        // 3. Parse Mode Byte
        // Controls 'out' argument usage
        uint8_t mode_byte = Data[offset++];

        // 4. Create Input Tensors
        // Note: torch::mm strictly requires 2D tensors.
        // mat1: (n x m)
        // mat2: (m x p)
        torch::Tensor mat1 = consume_tensor(Data, offset, Size, {n, m}, dtype);
        torch::Tensor mat2 = consume_tensor(Data, offset, Size, {m, p}, dtype);

        // 5. Invoke torch::mm
        bool use_out = (mode_byte & 0x01);
        if (use_out) {
            torch::Tensor out;
            // Toggle between providing a correctly sized out tensor and an incorrectly sized one
            // to test resizing logic or error handling.
            bool correct_shape = (mode_byte & 0x02);

            if (correct_shape) {
                out = torch::empty({n, p}, torch::TensorOptions().dtype(dtype));
            } else {
                // Some arbitrary shape, e.g., {p, n} or empty {0}
                if (mode_byte & 0x04) {
                    out = torch::empty({p, n}, torch::TensorOptions().dtype(dtype));
                } else {
                    out = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                }
            }

            // torch::mm has only the two-argument form; use explicit out variant.
            torch::mm_out(out, mat1, mat2);
        } else {
            torch::mm(mat1, mat2);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input for libfuzzer
    }
    return 0; // keep the input
}
