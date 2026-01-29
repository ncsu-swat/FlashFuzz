#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least 1 byte for number of types
        if (Size < 1) {
            return 0;
        }
        
        // Parse number of types to include in the list (1-10)
        uint8_t num_types = (Data[offset++] % 10) + 1;
        
        // Create a vector to hold the ScalarTypes
        std::vector<c10::ScalarType> type_list;
        
        // Parse each type from the input data
        for (uint8_t i = 0; i < num_types && offset < Size; ++i) {
            uint8_t type_selector = Data[offset++];
            c10::ScalarType dtype = fuzzer_utils::parseDataType(type_selector);
            type_list.push_back(dtype);
        }
        
        // Call promoteTypes pairwise to get the unified type
        if (type_list.size() > 0) {
            c10::ScalarType unified_type = type_list[0];
            for (size_t i = 1; i < type_list.size(); ++i) {
                try {
                    unified_type = c10::promoteTypes(unified_type, type_list[i]);
                } catch (...) {
                    // Some type combinations may not be promotable, silently skip
                }
            }
            
            // Create a tensor with the unified type to verify it works
            try {
                std::vector<int64_t> shape = {1, 1};
                auto options = torch::TensorOptions().dtype(unified_type);
                torch::Tensor test_tensor = torch::zeros(shape, options);
            } catch (...) {
                // Some types may not be valid for tensor creation
            }
        }
        
        // Try with mixed numeric and complex types
        if (offset + 1 < Size) {
            try {
                c10::ScalarType mixed_result = c10::promoteTypes(torch::kFloat, torch::kComplexFloat);
                (void)mixed_result;
            } catch (...) {
                // Silently handle
            }
        }
        
        // Try with potentially incompatible types
        if (offset + 1 < Size) {
            try {
                c10::ScalarType incompatible_result = c10::promoteTypes(torch::kBool, torch::kComplexDouble);
                (void)incompatible_result;
            } catch (...) {
                // Silently handle
            }
        }
        
        // Try with duplicate types
        if (offset < Size) {
            uint8_t type_selector = Data[offset++];
            c10::ScalarType dtype = fuzzer_utils::parseDataType(type_selector);
            try {
                c10::ScalarType duplicate_result = c10::promoteTypes(dtype, dtype);
                (void)duplicate_result;
            } catch (...) {
                // Silently handle
            }
        }
        
        // Try with all supported types
        if (offset < Size && Data[offset++] % 5 == 0) {
            std::vector<c10::ScalarType> all_types = {
                torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
                torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64,
                torch::kBool, torch::kComplexFloat, torch::kComplexDouble
            };
            c10::ScalarType all_result = all_types[0];
            for (size_t i = 1; i < all_types.size(); ++i) {
                try {
                    all_result = c10::promoteTypes(all_result, all_types[i]);
                } catch (...) {
                    // Silently handle promotion failures
                }
            }
            (void)all_result;
        }
        
        // Additional test: result_type which is another type unification function
        if (offset + 2 < Size) {
            uint8_t type1_selector = Data[offset++];
            uint8_t type2_selector = Data[offset++];
            c10::ScalarType dtype1 = fuzzer_utils::parseDataType(type1_selector);
            c10::ScalarType dtype2 = fuzzer_utils::parseDataType(type2_selector);
            
            try {
                // Create tensors and use torch::result_type
                auto t1 = torch::zeros({1}, torch::TensorOptions().dtype(dtype1));
                auto t2 = torch::zeros({1}, torch::TensorOptions().dtype(dtype2));
                c10::ScalarType result = torch::result_type(t1, t2);
                (void)result;
            } catch (...) {
                // Silently handle
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