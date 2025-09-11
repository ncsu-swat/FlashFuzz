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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a Storage for double type
        at::Storage storage;
        
        // Try different ways to create/use Storage based on remaining data
        if (offset < Size) {
            uint8_t option = Data[offset++] % 4;
            
            switch (option) {
                case 0: {
                    // Create empty storage
                    storage = at::Storage(at::caffe2::TypeMeta::Make<double>());
                    break;
                }
                case 1: {
                    // Create storage with size
                    int64_t size = tensor.numel() > 0 ? tensor.numel() : 1;
                    storage = at::Storage(at::caffe2::TypeMeta::Make<double>(), size, at::DataPtr(nullptr, at::Device(at::DeviceType::CPU)));
                    break;
                }
                case 2: {
                    // Create storage from tensor data
                    // Convert tensor to double type if needed
                    torch::Tensor doubleTensor = tensor.to(torch::kDouble);
                    storage = doubleTensor.storage();
                    break;
                }
                case 3: {
                    // Create storage with data from raw bytes
                    size_t remaining = Size - offset;
                    size_t num_doubles = remaining / sizeof(double);
                    
                    if (num_doubles > 0) {
                        std::vector<double> values(num_doubles);
                        std::memcpy(values.data(), Data + offset, num_doubles * sizeof(double));
                        torch::Tensor temp = torch::from_blob(values.data(), {static_cast<int64_t>(num_doubles)}, torch::kDouble).clone();
                        storage = temp.storage();
                    } else {
                        storage = at::Storage(at::caffe2::TypeMeta::Make<double>());
                    }
                    break;
                }
            }
        } else {
            // Default case if we've consumed all data
            storage = at::Storage(at::caffe2::TypeMeta::Make<double>());
        }
        
        // Test storage operations
        if (offset < Size && storage.nbytes() > 0) {
            uint8_t op = Data[offset++] % 4;
            
            switch (op) {
                case 0: {
                    // Resize storage
                    int64_t new_size = (offset < Size) ? Data[offset++] : 10;
                    storage.resize_(new_size * sizeof(double));
                    break;
                }
                case 1: {
                    // Fill storage with value
                    double fill_value = 0.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&fill_value, Data + offset, sizeof(double));
                        offset += sizeof(double);
                    }
                    if (storage.data_ptr().get() != nullptr) {
                        double* data_ptr = static_cast<double*>(storage.data_ptr().get());
                        size_t num_elements = storage.nbytes() / sizeof(double);
                        for (size_t i = 0; i < num_elements; i++) {
                            data_ptr[i] = fill_value;
                        }
                    }
                    break;
                }
                case 2: {
                    // Access elements
                    if (storage.data_ptr().get() != nullptr) {
                        double* data_ptr = static_cast<double*>(storage.data_ptr().get());
                        size_t num_elements = storage.nbytes() / sizeof(double);
                        for (size_t i = 0; i < num_elements; i++) {
                            double val = data_ptr[i];
                            data_ptr[i] = val * 2.0;
                        }
                    }
                    break;
                }
                case 3: {
                    // Copy storage
                    at::Storage copy_storage = storage.copy();
                    break;
                }
            }
        }
        
        // Create a tensor from the storage
        if (storage.nbytes() > 0) {
            std::vector<int64_t> sizes;
            if (tensor.dim() > 0) {
                sizes = tensor.sizes().vec();
            } else {
                sizes = {static_cast<int64_t>(storage.nbytes() / sizeof(double))};
            }
            
            // Ensure product of sizes matches storage size
            int64_t total_size = 1;
            for (auto s : sizes) {
                total_size *= s;
            }
            
            if (total_size * sizeof(double) > storage.nbytes()) {
                sizes = {static_cast<int64_t>(storage.nbytes() / sizeof(double))};
            }
            
            torch::Tensor result = torch::from_blob(
                storage.data_ptr().get(),
                sizes,
                torch::TensorOptions().dtype(torch::kDouble)
            ).clone();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
