#include "fuzzer_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <string_view>
#include <filesystem>
#include <thread> // If needed for future multi-threading utilities

namespace
{ // Keep internal helpers in anonymous namespace

    // Using C++17's string_view for constant strings.
    constexpr std::string_view separator = "\n-----\n";

    // Supported types definition (internal to this compilation unit)
    const std::vector<torch::ScalarType> supported_types = {
        torch::kFloat, torch::kDouble, torch::kHalf, torch::kBFloat16,
        torch::kComplexFloat, torch::kComplexDouble,
        torch::kInt8, torch::kUInt8, torch::kInt16, torch::kInt32, torch::kInt64,
        torch::kBool
        // torch::kFloat32 is same as kFloat, removed duplicate
    };

} // namespace

namespace fuzzer_utils
{

    // --- Timestamp & Logging Implementation ---

    bool ensure_log_directory_exists(const std::string &dir)
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (!fs::exists(dir, ec))
        {
            if (!fs::create_directories(dir, ec) && ec)
            {
                std::cerr << "Warning: Could not create log directory '" << dir << "': " << ec.message() << std::endl;
                return false;
            }
        }
        else if (!fs::is_directory(dir, ec) && !ec)
        {
            std::cerr << "Warning: Path '" << dir << "' exists but is not a directory." << std::endl;
            return false;
        }
        else if (ec)
        {
            std::cerr << "Warning: Filesystem error checking directory '" << dir << "': " << ec.message() << std::endl;
            return false;
        }
        return true;
    }

    std::string currentTimestamp()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto time = system_clock::to_time_t(now);
        std::tm tm;
#if defined(_MSC_VER)
        localtime_s(&tm, &time);
#else
        localtime_r(&time, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    std::string sanitizedTimestamp()
    {
        std::string ts = currentTimestamp();
        std::replace(ts.begin(), ts.end(), ' ', '_');
        std::replace(ts.begin(), ts.end(), ':', '-'); // Replace colon with hyphen for broader filesystem compatibility
        return ts;
    }

    void logErrorMessage(const std::string &msg)
    {
        if (!ensure_log_directory_exists())
            return;
        std::ofstream ofs("error.log", std::ios::app);
        if (!ofs)
        {
            std::cerr << "Warning: Unable to open error.log to log error messages." << std::endl;
            return;
        }
        ofs << "Timestamp: " << currentTimestamp() << "\n"
            << msg << separator;
    }

    void saveErrorInput(const uint8_t *data, size_t size)
    {
        if (!ensure_log_directory_exists())
            return;
        std::ofstream ofs("error_inputs.log", std::ios::app | std::ios::binary);
        if (!ofs)
        {
            std::cerr << "Warning: Unable to open error_inputs.log to log error input." << std::endl;
            return;
        }
        ofs << "Timestamp: " << currentTimestamp() << "\n";
        ofs.write(reinterpret_cast<const char *>(data), size);
        ofs << separator;
    }

    void saveDiffInput(const uint8_t *data, size_t size, const std::string &timestamp)
    {
        if (!ensure_log_directory_exists())
            return;
        std::ofstream ofs("diff_inputs.log", std::ios::app | std::ios::binary);
        if (!ofs)
        {
            std::cerr << "Warning: Unable to open diff_inputs.log to log difference input." << std::endl;
            return;
        }
        // Fix for line 123: No change needed, variable name now matches
        ofs << "Timestamp: " << timestamp << "\n";
        ofs.write(reinterpret_cast<const char *>(data), size);
        ofs << separator;
    }

    // --- Tensor Parsing Implementation ---

    torch::ScalarType parseDataType(uint8_t selector)
    {
        size_t index = selector % supported_types.size();
#ifdef DEBUG_FUZZ
        std::cout << "Data type selector: " << static_cast<int>(selector) << ", index: " << index << ", type: " << c10::toString(supported_types[index]) << std::endl;
#endif
        return supported_types[index];
    }

    uint8_t parseRank(uint8_t byte)
    {
        constexpr uint8_t range = MAX_RANK - MIN_RANK + 1;
        uint8_t rank = byte % range + MIN_RANK;
#ifdef DEBUG_FUZZ
        std::cout << "Rank byte: " << static_cast<int>(byte) << ", rank: " << static_cast<int>(rank) << std::endl;
#endif
        return rank;
    }

    std::vector<int64_t> parseShape(const uint8_t *data, size_t &offset, size_t size, uint8_t rank)
    {
        if (rank < MIN_RANK || rank > MAX_RANK)
        {
            throw std::runtime_error("Invalid rank specified for shape parsing.");
        }
        if (rank == 0)
        {
#ifdef DEBUG_FUZZ
            std::cout << "Parsing shape for rank 0 tensor." << std::endl;
#endif
            return {}; // Scalar tensor has empty shape
        }

        std::vector<int64_t> shape;
        shape.reserve(rank);
        const size_t bytes_needed_per_dim = sizeof(int64_t); // Using 8 bytes per dimension

#ifdef DEBUG_FUZZ
        std::cout << "Parsing shape for rank " << static_cast<int>(rank) << ", need " << rank * bytes_needed_per_dim << " bytes for dims." << std::endl;
        std::cout << "Current offset: " << offset << ", total size: " << size << std::endl;
#endif

        for (uint8_t i = 0; i < rank; ++i)
        {
            if (offset + bytes_needed_per_dim <= size)
            {
                int64_t dim_raw;
                std::memcpy(&dim_raw, data + offset, bytes_needed_per_dim);
                offset += bytes_needed_per_dim;

                // Make dimension deterministic and within bounds [MIN_TENSOR_SHAPE_DIMS, MAX_TENSOR_SHAPE_DIMS]
                // Using unsigned ensures non-negativity after potential abs, then modulo.
                uint64_t dim_abs = static_cast<uint64_t>(std::abs(dim_raw));
                int64_t dim = MIN_TENSOR_SHAPE_DIMS + static_cast<int64_t>(dim_abs % (MAX_TENSOR_SHAPE_DIMS - MIN_TENSOR_SHAPE_DIMS + 1));
                shape.push_back(dim);
#ifdef DEBUG_FUZZ
                std::cout << "  Dim " << static_cast<int>(i) << ": raw=" << dim_raw << ", abs=" << dim_abs << ", parsed=" << dim << std::endl;
#endif
            }
            else
            {
                // Not enough data, default to minimum dimension
                shape.push_back(MIN_TENSOR_SHAPE_DIMS);
#ifdef DEBUG_FUZZ
                std::cout << "  Dim " << static_cast<int>(i) << ": Not enough data, defaulting to " << MIN_TENSOR_SHAPE_DIMS << std::endl;
#endif
                // Update offset to prevent infinite loops if size is small but non-zero
                offset = size;
            }
        }

#ifdef DEBUG_FUZZ
        std::cout << "Parsed Shape: [";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << "], New offset: " << offset << std::endl;
#endif
        return shape;
    }

    std::vector<uint8_t> parseTensorData(const uint8_t *data, size_t &offset, size_t size,
                                         int64_t numElements, size_t dtypeSize)
    {
        // Check for potential overflow before calculating total bytes needed
        int64_t max_representable_elements = std::numeric_limits<size_t>::max() / (dtypeSize > 0 ? dtypeSize : 1);
        if (numElements < 0 || (dtypeSize > 0 && numElements > max_representable_elements))
        {
            throw std::runtime_error("Number of elements or dtypeSize leads to potential overflow or invalid.");
        }

        size_t totalBytesNeeded = static_cast<size_t>(numElements) * dtypeSize;
        std::vector<uint8_t> tensorData(totalBytesNeeded, 0); // Initialize with zeros

        size_t bytesAvailable = (offset < size) ? (size - offset) : 0;
        size_t bytesToCopy = std::min(totalBytesNeeded, bytesAvailable);

#ifdef DEBUG_FUZZ
        std::cout << "Parsing Tensor Data: numElements=" << numElements
                  << ", dtypeSize=" << dtypeSize << ", totalBytesNeeded=" << totalBytesNeeded
                  << ", offset=" << offset << ", size=" << size
                  << ", bytesAvailable=" << bytesAvailable << ", bytesToCopy=" << bytesToCopy << std::endl;
#endif

        if (bytesToCopy > 0)
        {
            std::memcpy(tensorData.data(), data + offset, bytesToCopy);
        }
        offset += bytesToCopy; // Advance offset by the number of bytes actually consumed

#ifdef DEBUG_FUZZ
        std::cout << "Finished parsing tensor data. New offset: " << offset << std::endl;
#endif

        return tensorData;
    }

    torch::Tensor createTensor(const uint8_t *Data, size_t Size, size_t &offset)
    {
        // Need at least 1 byte for dtype and 1 byte for rank.
        const size_t min_required_size = 2;
        if (Size < offset + min_required_size)
        {
            throw std::runtime_error("Input data too small for basic tensor metadata (dtype, rank). Offset: " + std::to_string(offset) + ", Size: " + std::to_string(Size));
        }

#ifdef DEBUG_FUZZ
        std::cout << "\n--- Creating Tensor ---" << std::endl;
        std::cout << "Initial offset: " << offset << ", Size: " << Size << std::endl;
#endif

        // 1. Parse Data Type
        const uint8_t dtype_selector = Data[offset++];
        const auto dtype = parseDataType(dtype_selector);
        const size_t dtypeSize = c10::elementSize(dtype); // Use c10::elementsize for robustness

        if (dtypeSize == 0 && dtype != torch::kBool)
        { // kBool size is 1 byte in memory typically
          // Potentially handle error or default for unknown/unsupported size 0 types if any exist beyond theoretical
          // For now, let it proceed, but be aware. torch::from_blob might handle it.
          // #ifdef DEBUG_FUZZ
          //             std::cout << "Warning: dtype " << c10::elementSize(dtype); << " has element size 0." << std::endl;
          // #endif
        }

        // 2. Parse Rank
        const uint8_t rank_byte = Data[offset++];
        const uint8_t rank = parseRank(rank_byte);

        // 3. Parse Shape
        // parseShape updates the offset internally
        auto shape = parseShape(Data, offset, Size, rank);

        // 4. Compute Number of Elements
        int64_t numElements = 1;
        bool shape_is_valid = true;
        if (rank > 0 && shape.empty())
        {
            // This case shouldn't happen if parseShape respects rank, but as a safeguard
            shape_is_valid = false;
#ifdef DEBUG_FUZZ
            std::cout << "Error: Rank > 0 but shape is empty after parsing." << std::endl;
#endif
        }
        for (const auto &dim : shape)
        {
            if (dim < 0)
            { // Dimensions must be non-negative
                shape_is_valid = false;
#ifdef DEBUG_FUZZ
                std::cout << "Error: Invalid negative dimension " << dim << " in shape." << std::endl;
#endif
                break;
            }
            // Check for potential overflow when calculating numElements
            if (dim > 0 && numElements > std::numeric_limits<int64_t>::max() / dim)
            {
                shape_is_valid = false;
#ifdef DEBUG_FUZZ
                std::cout << "Error: Potential overflow calculating number of elements." << std::endl;
#endif
                break; // Avoid overflow
            }
            numElements *= dim;
        }

        if (!shape_is_valid)
        {
            // Handle invalid shape - perhaps create a default small tensor or throw?
            // Throwing is safer for fuzzing as it signals a clear parsing issue.
            throw std::runtime_error("Parsed shape is invalid (negative dimension or overflow).");
        }

#ifdef DEBUG_FUZZ
        std::cout << "Calculated numElements: " << numElements << std::endl;
#endif

// Conditional compilation for random vs input-based tensor data
#if USE_RANDOM_TENSOR
#ifdef DEBUG_FUZZ
        std::cout << "Generating random tensor..." << std::endl;
#endif
        // Generate a random tensor
        // Ensure options specify the correct dtype
        auto options = torch::TensorOptions().dtype(dtype);
        return torch::rand(shape, options);
#else
        // 5. Parse Tensor Data
        auto inputBytes = parseTensorData(Data, offset, Size, numElements, dtypeSize);

        // 6. Create Tensor from Blob
        // Note: torch::from_blob requires a non-const pointer to data it does *not* own.
        // Cloning is essential to make PyTorch manage the memory.
        // Handle the edge case of a 0-element tensor correctly.
        torch::Tensor tensor;
        if (numElements == 0)
        {
            // Create an empty tensor with the correct shape and dtype
            auto options = torch::TensorOptions().dtype(dtype);
            // Using empty is generally preferred for creating tensors with shape but no data/storage
            tensor = torch::empty(shape, options);
#ifdef DEBUG_FUZZ
            std::cout << "Created empty tensor for numElements=0." << std::endl;
#endif
        }
        else if (inputBytes.empty() && numElements > 0)
        {
            // This might happen if parseTensorData couldn't read any bytes (e.g., offset >= Size)
            // Decide on behavior: throw, or create an uninitialized tensor?
            // Creating uninitialized might hide bugs, throwing is safer.
            // However, maybe the API call itself handles uninitialized data? For now, let's try creating it.
            auto options = torch::TensorOptions().dtype(dtype);
            tensor = torch::empty(shape, options); // Creates tensor with undefined data
#ifdef DEBUG_FUZZ
            std::cout << "Warning: Creating tensor with uninitialized data (inputBytes empty, numElements > 0)." << std::endl;
#endif
            // Alternatively, throw:
            // throw std::runtime_error("Not enough data to populate tensor elements.");
        }
        else
        {
            // Ensure the size of inputBytes matches expected size. If not, it implies parseTensorData hit EOF.
            // from_blob doesn't care if the data is partial, but downstream usage will likely fail.
            // The clone after from_blob makes a copy, potentially copying only partial data if inputBytes is short.
            auto options = torch::TensorOptions().dtype(dtype);
            // We pass inputBytes.data(), which points to the vector's buffer.
            // The lifetime of inputBytes must exceed the from_blob call. Cloning makes this safe.
            tensor = torch::from_blob(inputBytes.data(), shape, options).clone();
#ifdef DEBUG_FUZZ
            std::cout << "Created tensor from blob and cloned." << std::endl;
#endif
        }

#ifdef DEBUG_FUZZ
        std::cout << "--- Finished Creating Tensor ---" << std::endl;
        std::cout << "Final offset: " << offset << std::endl;
        std::cout << "Tensor details: " << tensor.sizes() << ", dtype=" << tensor.dtype() << ", is_contiguous=" << tensor.is_contiguous() << std::endl;
#endif
        return tensor;
#endif // USE_RANDOM_TENSOR
    }

    // --- Comparison (Optional) ---
    void compareTensors(const torch::Tensor &t1, const torch::Tensor &t2, const uint8_t *data, size_t size, double rtol, double atol)
    {
        // Check if shapes match
        if (t1.sizes() != t2.sizes())
        {
            std::cerr << "Shape mismatch: " << t1.sizes() << " vs " << t2.sizes() << std::endl;
        }

        // Check if data types match
        if (t1.dtype() != t2.dtype())
        {
            std::cerr << "Dtype mismatch: " << t1.dtype() << " vs " << t2.dtype() << std::endl;
        }

        bool is_close = torch::allclose(t1, t2, rtol, atol);

        if (!is_close)
        {
            // Find max absolute difference
            auto abs_diff = torch::abs(t1 - t2);
            auto max_diff = torch::max(abs_diff).item<float>();
            auto max_val = std::max(torch::max(torch::abs(t1)).item<float>(),
                                    torch::max(torch::abs(t2)).item<float>());

            std::cerr << "Max difference: " << max_diff << " (relative to max value: "
                      << (max_val > 0 ? max_diff / max_val : max_diff) << ")" << std::endl;

            // Save the input data that caused the discrepancy
            std::string timestamp = sanitizedTimestamp();

            // Ensure directory exists
            (void)ensure_log_directory_exists("diff_inputs");

            // Save the raw input data to a binary file
            std::string bin_filename = "diff_inputs/input_" + timestamp + ".bin";
            std::ofstream bin_file(bin_filename, std::ios::binary);
            if (bin_file)
            {
                bin_file.write(reinterpret_cast<const char *>(data), size);
                bin_file.close();
                std::cerr << "Saved input data to: " << bin_filename << std::endl;
            }
            else
            {
                std::cerr << "Failed to save input data to file" << std::endl;
            }
        }
    }
} // namespace fuzzer_utils
