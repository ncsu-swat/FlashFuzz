#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // softmax requires at least 1 dimension
        if (input.dim() == 0) {
            return 0;
        }
        
        if (offset >= Size) {
            return 0;
        }
        
        // Get dimension from fuzzer data, valid range is [-dim, dim-1]
        uint8_t dim_byte = Data[offset++];
        int64_t dim = static_cast<int64_t>(dim_byte % input.dim());
        
        // Test basic softmax
        torch::Tensor output = torch::softmax(input, dim);
        
        // Test softmax with different dtypes based on fuzzer input
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            try {
                if (dtype_selector == 0) {
                    output = torch::softmax(input, dim, torch::kFloat);
                } else if (dtype_selector == 1) {
                    output = torch::softmax(input, dim, torch::kDouble);
                } else if (dtype_selector == 2) {
                    // Test with negative dimension
                    int64_t neg_dim = -(static_cast<int64_t>(dim_byte % input.dim()) + 1);
                    output = torch::softmax(input, neg_dim);
                } else {
                    // Test with the original computed dimension
                    output = torch::softmax(input, dim);
                }
            } catch (const std::exception &) {
                // Some dtype conversions may fail, that's expected
            }
        }
        
        // Additional coverage: test with different input types
        if (offset < Size) {
            uint8_t input_type = Data[offset++] % 3;
            try {
                if (input_type == 0) {
                    // Test with float input
                    torch::Tensor float_input = input.to(torch::kFloat);
                    output = torch::softmax(float_input, dim);
                } else if (input_type == 1) {
                    // Test with double input
                    torch::Tensor double_input = input.to(torch::kDouble);
                    output = torch::softmax(double_input, dim);
                } else {
                    // Test with contiguous input
                    torch::Tensor contig_input = input.contiguous();
                    output = torch::softmax(contig_input, dim);
                }
            } catch (const std::exception &) {
                // Type conversion failures are expected for some inputs
            }
        }
        
        // Verify output properties (softmax should sum to 1 along dim)
        // This exercises the output tensor
        (void)output.sum(dim);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}