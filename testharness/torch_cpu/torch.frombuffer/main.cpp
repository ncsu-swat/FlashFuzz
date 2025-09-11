#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a meaningful buffer
        if (Size < 4) {
            return 0;
        }
        
        // Parse buffer size and offset
        uint32_t buffer_size = 0;
        if (offset + sizeof(buffer_size) <= Size) {
            std::memcpy(&buffer_size, Data + offset, sizeof(buffer_size));
            offset += sizeof(buffer_size);
            
            // Limit buffer size to avoid excessive memory usage
            buffer_size = buffer_size % 1024;
        }
        
        // Create a buffer to use with frombuffer
        std::vector<uint8_t> buffer;
        buffer.reserve(buffer_size);
        
        // Fill buffer with data from input
        for (uint32_t i = 0; i < buffer_size && offset < Size; i++) {
            buffer.push_back(Data[offset++]);
        }
        
        // Parse data type for the buffer
        torch::ScalarType dtype = torch::kUInt8;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Parse count (number of elements)
        int64_t count = -1;  // -1 means use all buffer
        if (offset + sizeof(count) <= Size) {
            std::memcpy(&count, Data + offset, sizeof(count));
            offset += sizeof(count);
        }
        
        // Parse byte offset within the buffer
        int64_t byte_offset = 0;
        if (offset + sizeof(byte_offset) <= Size) {
            std::memcpy(&byte_offset, Data + offset, sizeof(byte_offset));
            offset += sizeof(byte_offset);
        }
        
        // Parse require_writable flag
        bool require_writable = false;
        if (offset < Size) {
            require_writable = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Try different combinations of frombuffer parameters
        try {
            // Basic usage with all parameters
            torch::Tensor tensor1 = torch::from_blob(
                buffer.data(),
                {static_cast<int64_t>(buffer.size())},
                torch::TensorOptions().dtype(dtype)
            );
            
            // Use with count parameter
            if (!buffer.empty()) {
                torch::Tensor tensor2 = torch::from_blob(
                    buffer.data(),
                    {count > 0 ? std::min(count, static_cast<int64_t>(buffer.size())) : static_cast<int64_t>(buffer.size())},
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            // Use with byte_offset
            if (byte_offset >= 0 && byte_offset < static_cast<int64_t>(buffer.size())) {
                torch::Tensor tensor3 = torch::from_blob(
                    buffer.data() + byte_offset,
                    {static_cast<int64_t>(buffer.size()) - byte_offset},
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            // Try with empty buffer
            std::vector<uint8_t> empty_buffer;
            if (buffer.empty()) {
                torch::Tensor tensor4 = torch::from_blob(
                    empty_buffer.data(),
                    {0},
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            // Try with different shapes
            if (buffer.size() >= 4) {
                torch::Tensor tensor5 = torch::from_blob(
                    buffer.data(),
                    {2, 2},
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            if (buffer.size() >= 8) {
                torch::Tensor tensor6 = torch::from_blob(
                    buffer.data(),
                    {2, 2, 2},
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            // Try with different strides
            if (buffer.size() >= 4) {
                std::vector<int64_t> sizes = {2, 2};
                std::vector<int64_t> strides = {2, 1};
                torch::Tensor tensor7 = torch::from_blob(
                    buffer.data(),
                    sizes,
                    strides,
                    torch::TensorOptions().dtype(dtype)
                );
            }
            
            // Try with require_writable
            if (!buffer.empty()) {
                torch::Tensor tensor8 = torch::from_blob(
                    buffer.data(),
                    {static_cast<int64_t>(buffer.size())},
                    [&buffer](void*) {}, // Deleter function
                    torch::TensorOptions().dtype(dtype)
                );
            }
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and part of testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
