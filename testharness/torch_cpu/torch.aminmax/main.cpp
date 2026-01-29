#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with tuple result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // aminmax doesn't work on complex types
        if (input.is_complex()) {
            return 0;
        }
        
        // Get a dimension to use for aminmax if needed
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Get keepdim flag
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Variant 1: aminmax without dimension (returns min/max over all elements)
        // This only works on non-empty tensors
        if (input.numel() > 0) {
            auto result1 = torch::aminmax(input);
            // Access the results to ensure they are computed
            auto min_val = std::get<0>(result1);
            auto max_val = std::get<1>(result1);
            (void)min_val;
            (void)max_val;
        }
        
        // Variant 2: aminmax with dimension
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                auto result2 = torch::aminmax(input, dim, keepdim);
                auto min_val = std::get<0>(result2);
                auto max_val = std::get<1>(result2);
                (void)min_val;
                (void)max_val;
            } catch (const std::exception&) {
                // Some shape combinations may fail
            }
        }
        
        // Variant 3: out variant with dimension
        if (input.dim() > 0 && input.numel() > 0) {
            try {
                // Compute expected output shape
                std::vector<int64_t> out_shape;
                for (int64_t i = 0; i < input.dim(); i++) {
                    if (i == dim) {
                        if (keepdim) {
                            out_shape.push_back(1);
                        }
                        // If not keepdim, skip this dimension
                    } else {
                        out_shape.push_back(input.size(i));
                    }
                }
                
                // Handle edge case where output would be a scalar
                if (out_shape.empty()) {
                    out_shape.push_back(1);
                }
                
                torch::Tensor min_out = torch::empty(out_shape, input.options());
                torch::Tensor max_out = torch::empty(out_shape, input.options());
                
                torch::aminmax_out(min_out, max_out, input, dim, keepdim);
            } catch (const std::exception&) {
                // Ignore exceptions from out variant - shape mismatches expected
            }
        }
        
        // Variant 4: Test with different data types
        if (offset < Size && input.numel() > 0) {
            uint8_t dtype_selector = Data[offset++] % 4;
            try {
                torch::Tensor converted;
                switch (dtype_selector) {
                    case 0:
                        converted = input.to(torch::kFloat32);
                        break;
                    case 1:
                        converted = input.to(torch::kFloat64);
                        break;
                    case 2:
                        converted = input.to(torch::kInt32);
                        break;
                    case 3:
                        converted = input.to(torch::kInt64);
                        break;
                }
                auto result = torch::aminmax(converted);
                (void)result;
            } catch (const std::exception&) {
                // Type conversion might fail
            }
        }
        
        // Variant 5: Test with contiguous vs non-contiguous tensor
        if (input.dim() >= 2 && input.numel() > 0) {
            try {
                auto transposed = input.transpose(0, 1);
                auto result = torch::aminmax(transposed);
                (void)result;
            } catch (const std::exception&) {
                // Ignore failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}