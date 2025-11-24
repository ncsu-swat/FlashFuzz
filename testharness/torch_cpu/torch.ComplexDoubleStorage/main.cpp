#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>      // For std::min
#include <cstring>        // For std::memcpy
#include <cstdlib>        // For std::abs
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Target API: torch.ComplexDoubleStorage
        // Create a tensor with complex double values
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to complex double if it's not already
        if (tensor.dtype() != torch::kComplexDouble) {
            tensor = tensor.to(torch::kComplexDouble);
        }
        
        // Get the storage from the tensor
        auto storage = tensor.storage();
        
        // Test various storage operations
        size_t storage_size = storage.nbytes() / sizeof(c10::complex<double>);
        size_t bounded_storage_size = std::min<size_t>(storage_size, 1024);
        
        // Test data access if storage is not empty
        if (bounded_storage_size > 0) {
            auto typed_data_ptr = static_cast<const c10::complex<double>*>(storage.data());
            if (typed_data_ptr) {
                auto first_element = typed_data_ptr[0];
                (void)first_element;
            }
            
            // Test copy constructor
            c10::Storage storage_copy = storage;
            
            // Test move constructor if we have enough data
            if (offset + 1 < Size) {
                uint8_t move_flag = Data[offset++];
                if (move_flag % 2 == 0) {
                    c10::Storage storage_moved = std::move(storage_copy);
                }
            }
            
            // Test fill operation by modifying data directly
            if (offset + sizeof(c10::complex<double>) <= Size) {
                c10::complex<double> fill_value;
                std::memcpy(&fill_value, Data + offset, sizeof(c10::complex<double>));
                offset += sizeof(c10::complex<double>);
                
                auto typed_ptr = static_cast<c10::complex<double>*>(storage.mutable_data());
                if (typed_ptr) {
                    for (size_t i = 0; i < bounded_storage_size; ++i) {
                        typed_ptr[i] = fill_value;
                    }
                }
            }
            
            // Test set operation
            if (bounded_storage_size > 1 && offset + sizeof(c10::complex<double>) <= Size) {
                c10::complex<double> set_value;
                std::memcpy(&set_value, Data + offset, sizeof(c10::complex<double>));
                offset += sizeof(c10::complex<double>);
                
                size_t index = 1; // Use second element to avoid potential issues with first element
                auto typed_ptr = static_cast<c10::complex<double>*>(storage.mutable_data());
                if (typed_ptr) {
                    typed_ptr[index] = set_value;
                    auto retrieved_value = typed_ptr[index];
                    (void)retrieved_value;
                }
            }
            
            // Test swap operation
            if (offset + 1 < Size) {
                uint8_t swap_flag = Data[offset++];
                if (swap_flag % 2 == 0) {
                    c10::Allocator* allocator = storage.allocator();
                    if (!allocator) {
                        allocator = c10::GetAllocator(storage.device().type());
                    }
                    size_t alloc_bytes = bounded_storage_size * sizeof(c10::complex<double>);
                    c10::Storage another_storage(
                        c10::Storage::use_byte_size_t(),
                        alloc_bytes,
                        allocator,
                        /*resizable=*/true);
                    std::swap(storage, another_storage);
                }
            }
        }
        
        // Create a new storage with specific size
        if (offset + sizeof(int64_t) <= Size) {
            int64_t explicit_size;
            std::memcpy(&explicit_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Bound the size to avoid excessive memory usage
            explicit_size = std::abs(explicit_size) % 1000;
            
            size_t explicit_bytes = static_cast<size_t>(explicit_size) * sizeof(c10::complex<double>);
            c10::Storage explicit_storage(
                c10::Storage::use_byte_size_t(),
                explicit_bytes,
                c10::GetAllocator(c10::kCPU),
                /*resizable=*/true);
            
            // Test empty storage creation
            c10::Storage empty_storage;
            
            // Test storage from data pointer
            if (explicit_size > 0) {
                std::vector<c10::complex<double>> data_vec(explicit_size);
                c10::Storage data_storage(
                    c10::Storage::use_byte_size_t(),
                    explicit_bytes,
                    c10::GetAllocator(c10::kCPU),
                    /*resizable=*/true);
                (void)data_vec;
            }
        }
        
        // Test storage creation from vector
        if (offset + 1 < Size) {
            uint8_t vec_size = Data[offset++];
            vec_size = vec_size % 100; // Limit vector size
            
            std::vector<c10::complex<double>> vec(vec_size);
            for (size_t i = 0; i < vec_size && offset + sizeof(c10::complex<double>) <= Size; i++) {
                std::memcpy(&vec[i], Data + offset, sizeof(c10::complex<double>));
                offset += sizeof(c10::complex<double>);
            }
            
            size_t vec_bytes = static_cast<size_t>(vec_size) * sizeof(c10::complex<double>);
            c10::Storage vec_storage(
                c10::Storage::use_byte_size_t(),
                vec_bytes,
                c10::GetAllocator(c10::kCPU),
                /*resizable=*/true);
            (void)vec_storage;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
