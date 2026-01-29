#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

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
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Determine number of tensors to create (2-4 for meaningful dstack)
        uint8_t num_tensors = (Data[offset++] % 3) + 2;

        // Parse common dimensions for compatible tensors
        // dstack stacks along the third dimension, so dims 0 and 1 must match
        uint8_t dim0 = (Data[offset++] % 4) + 1;  // 1-4
        uint8_t dim1 = (Data[offset++] % 4) + 1;  // 1-4

        // Parse dtype
        auto dtype = fuzzer_utils::parseDataType(Data[offset++]);

        // Create a vector to hold our tensors with compatible shapes
        std::vector<torch::Tensor> tensors;

        for (uint8_t i = 0; i < num_tensors && offset < Size; ++i) {
            // Vary the third dimension (or create 1D/2D tensors that will be broadcast)
            uint8_t tensor_type = Data[offset++] % 4;

            torch::Tensor tensor;
            try {
                if (tensor_type == 0) {
                    // 1D tensor (will be reshaped to 1 x 1 x len by dstack)
                    int64_t len = dim0;
                    tensor = torch::randn({len}, torch::TensorOptions().dtype(dtype));
                } else if (tensor_type == 1) {
                    // 2D tensor (will be reshaped to dim0 x dim1 x 1 by dstack)
                    tensor = torch::randn({dim0, dim1}, torch::TensorOptions().dtype(dtype));
                } else if (tensor_type == 2) {
                    // 3D tensor with varying depth
                    uint8_t dim2 = (offset < Size) ? (Data[offset++] % 4) + 1 : 1;
                    tensor = torch::randn({dim0, dim1, dim2}, torch::TensorOptions().dtype(dtype));
                } else {
                    // 3D tensor with depth 1
                    tensor = torch::randn({dim0, dim1, 1}, torch::TensorOptions().dtype(dtype));
                }
                tensors.push_back(tensor);
            } catch (...) {
                // Silently skip if tensor creation fails
                continue;
            }
        }

        if (tensors.empty()) {
            return 0;
        }

        // Test 1: dstack with single tensor
        if (tensors.size() >= 1) {
            try {
                torch::Tensor result = torch::dstack({tensors[0]});
                (void)result;
            } catch (...) {
                // Shape issues possible, silently continue
            }
        }

        // Test 2: dstack with multiple tensors of same base shape
        if (tensors.size() >= 2) {
            // Create tensors with guaranteed compatible shapes
            std::vector<torch::Tensor> compatible_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                // Reshape all to 2D with same dim0 x dim1, let dstack handle the rest
                try {
                    torch::Tensor t = torch::randn({dim0, dim1}, torch::TensorOptions().dtype(dtype));
                    compatible_tensors.push_back(t);
                } catch (...) {
                    continue;
                }
            }
            if (compatible_tensors.size() >= 2) {
                try {
                    torch::Tensor result = torch::dstack(compatible_tensors);
                    (void)result;
                } catch (...) {
                    // Silently handle
                }
            }
        }

        // Test 3: dstack with 1D tensors (special case - treated as 1x1xN after broadcast)
        if (offset + 2 < Size) {
            int64_t len = (Data[offset++] % 8) + 1;
            std::vector<torch::Tensor> tensors_1d;
            for (int i = 0; i < 3; ++i) {
                tensors_1d.push_back(torch::randn({len}, torch::TensorOptions().dtype(dtype)));
            }
            try {
                torch::Tensor result = torch::dstack(tensors_1d);
                (void)result;
            } catch (...) {
                // Silently handle
            }
        }

        // Test 4: dstack with 3D tensors of same dim0, dim1 but different dim2
        if (offset + 4 < Size) {
            std::vector<torch::Tensor> tensors_3d;
            for (int i = 0; i < 3 && offset < Size; ++i) {
                int64_t depth = (Data[offset++] % 5) + 1;
                tensors_3d.push_back(torch::randn({dim0, dim1, depth}, torch::TensorOptions().dtype(dtype)));
            }
            if (tensors_3d.size() >= 2) {
                try {
                    torch::Tensor result = torch::dstack(tensors_3d);
                    (void)result;
                } catch (...) {
                    // Silently handle
                }
            }
        }

        // Test 5: Use the original fuzzed tensors directly
        try {
            torch::Tensor result = torch::dstack(tensors);
            (void)result;
        } catch (...) {
            // Shape mismatch expected, silently continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}