#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to extract values for LongStorage
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create LongStorage from tensor data
        try {
            // Method 1: Create LongStorage directly using std::vector
            c10::IntArrayRef sizes = tensor.sizes();
            std::vector<int64_t> storage_vec(sizes.begin(), sizes.end());
            
            // Test basic properties
            size_t size = storage_vec.size();
            
            // Access elements (if any)
            if (size > 0) {
                int64_t first_element = storage_vec[0];
                
                // Test copy constructor
                std::vector<int64_t> storage_copy(storage_vec);
                
                // Test assignment operator
                std::vector<int64_t> storage_assigned = storage_vec;
                
                // Test equality
                bool equal = (storage_vec == storage_copy);
                bool not_equal = (storage_vec != storage_assigned);
            }
            
            // Create empty storage
            std::vector<int64_t> empty_storage;
            
            // Create storage with specific size
            if (offset < Size) {
                uint8_t storage_size = Data[offset++] % 10; // Limit to reasonable size
                std::vector<int64_t> sized_storage(storage_size);
                
                // Fill with values
                for (int i = 0; i < storage_size && offset + sizeof(int64_t) <= Size; i++) {
                    int64_t value;
                    std::memcpy(&value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    sized_storage[i] = value;
                }
            }
            
            // Method 2: Create from vector
            if (offset + 1 < Size) {
                uint8_t vec_size = Data[offset++] % 8; // Limit vector size
                std::vector<int64_t> vec;
                
                for (int i = 0; i < vec_size && offset + sizeof(int64_t) <= Size; i++) {
                    int64_t value;
                    std::memcpy(&value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    vec.push_back(value);
                }
                
                std::vector<int64_t> vec_storage(vec);
            }
            
            // Method 3: Create from initializer list
            if (offset + 3 < Size) {
                int64_t val1, val2, val3;
                std::memcpy(&val1, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&val2, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                std::memcpy(&val3, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                std::vector<int64_t> init_storage = {val1, val2, val3};
            }
            
            // Test resizing
            if (offset < Size && !storage_vec.empty()) {
                uint8_t new_size = Data[offset++] % 20;
                storage_vec.resize(new_size);
            }
        }
        catch (const c10::Error &e) {
            // PyTorch specific errors are expected and handled
        }
        
        // Try creating a tensor using vector for sizes
        if (offset + 2 < Size) {
            uint8_t storage_size = Data[offset++] % 5 + 1; // 1-5 dimensions
            std::vector<int64_t> dim_storage(storage_size);
            
            // Fill with dimension values
            for (int i = 0; i < storage_size && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure positive dimensions for tensor creation
                dim_storage[i] = std::abs(dim_value) % 100 + 1;
            }
            
            try {
                // Create tensor with dimensions from vector
                torch::Tensor result_tensor = torch::zeros(dim_storage);
            }
            catch (const c10::Error &e) {
                // Expected for invalid dimensions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}