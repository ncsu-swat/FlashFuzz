#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Extract parameters for linspace
        double start = 0.0;
        double end = 1.0;
        int64_t steps = 100;
        
        // Parse start value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse end value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse steps
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&steps, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Allow negative steps to test error handling
            // Don't clamp to positive values to test error cases
        }
        
        // Parse dtype
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Try different variants of linspace
        
        // Basic linspace
        torch::Tensor result1 = torch::linspace(start, end, steps);
        
        // Linspace with specified dtype
        torch::Tensor result2 = torch::linspace(start, end, steps, 
                                               torch::TensorOptions().dtype(dtype));
        
        // Linspace with layout and device options
        torch::Tensor result3 = torch::linspace(start, end, steps, 
                                               torch::TensorOptions()
                                               .dtype(dtype)
                                               .layout(torch::kStrided));
        
        // Edge case: steps = 0 or 1
        int64_t edge_steps = 0;
        if (offset < Size) {
            edge_steps = Data[offset++] % 2; // Either 0 or 1
            torch::Tensor result_edge = torch::linspace(start, end, edge_steps);
        }
        
        // Edge case: start = end
        torch::Tensor result_same = torch::linspace(start, start, steps);
        
        // Edge case: NaN or Inf values
        if (offset < Size) {
            uint8_t special_case = Data[offset++] % 4;
            double special_start = start;
            double special_end = end;
            
            if (special_case == 0) {
                special_start = std::numeric_limits<double>::quiet_NaN();
            } else if (special_case == 1) {
                special_end = std::numeric_limits<double>::quiet_NaN();
            } else if (special_case == 2) {
                special_start = std::numeric_limits<double>::infinity();
            } else if (special_case == 3) {
                special_end = std::numeric_limits<double>::infinity();
            }
            
            torch::Tensor result_special = torch::linspace(special_start, special_end, steps);
        }
        
        // Test with extremely large values
        if (offset < Size) {
            uint8_t large_case = Data[offset++] % 3;
            double large_start = start;
            double large_end = end;
            int64_t large_steps = steps;
            
            if (large_case == 0) {
                large_start = std::numeric_limits<double>::max() / 2;
                large_end = std::numeric_limits<double>::max();
            } else if (large_case == 1) {
                large_start = -std::numeric_limits<double>::max();
                large_end = std::numeric_limits<double>::max();
            } else if (large_case == 2) {
                large_steps = std::numeric_limits<int32_t>::max();
            }
            
            try {
                torch::Tensor result_large = torch::linspace(large_start, large_end, large_steps);
            } catch (const std::exception& e) {
                // Expected exception for very large steps
            }
        }
        
        // Test with extremely small values
        if (offset < Size) {
            uint8_t small_case = Data[offset++] % 2;
            double small_start = start;
            double small_end = end;
            
            if (small_case == 0) {
                small_start = std::numeric_limits<double>::min();
                small_end = std::numeric_limits<double>::min() * 10;
            } else if (small_case == 1) {
                small_start = -std::numeric_limits<double>::min();
                small_end = std::numeric_limits<double>::min();
            }
            
            torch::Tensor result_small = torch::linspace(small_start, small_end, steps);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
