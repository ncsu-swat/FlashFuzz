#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <fstream>        // For ofstream
#include <filesystem>     // For filesystem operations
#include <cstdlib>        // For getenv

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to test with
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various filesystem functions using std::filesystem
        
        // Test getenv
        if (offset < Size) {
            uint8_t env_selector = Data[offset++];
            std::vector<std::string> env_vars = {"PATH", "HOME", "USER", "TEMP", "PWD", "SHELL"};
            std::string env_var = env_vars[env_selector % env_vars.size()];
            const char* env_value = std::getenv(env_var.c_str());
        }
        
        // Test makedirs
        if (offset + 2 < Size) {
            uint8_t dir_selector = Data[offset++];
            uint8_t exist_ok_selector = Data[offset++];
            
            std::vector<std::string> test_dirs = {
                "test_dir", "test/nested/dir", "./tmp_dir", "/tmp/test_dir"
            };
            
            std::string dir_name = test_dirs[dir_selector % test_dirs.size()];
            bool exist_ok = (exist_ok_selector % 2 == 0);
            
            try {
                if (exist_ok) {
                    std::filesystem::create_directories(dir_name);
                } else {
                    std::filesystem::create_directory(dir_name);
                }
            } catch (const std::exception& e) {
                // Permissions or other issues might cause this to fail, which is fine
            }
        }
        
        // Test remove
        if (offset + 1 < Size) {
            uint8_t file_selector = Data[offset++];
            std::vector<std::string> test_files = {
                "test_file.txt", "nonexistent_file.txt", "./tmp_file.txt"
            };
            
            std::string file_name = test_files[file_selector % test_files.size()];
            
            // Create a file first to potentially remove
            if (file_selector % 2 == 0) {
                std::ofstream file(file_name);
                file << "test";
                file.close();
            }
            
            try {
                std::filesystem::remove(file_name);
            } catch (const std::exception& e) {
                // File might not exist, which is fine
            }
        }
        
        // Test rename
        if (offset + 2 < Size) {
            uint8_t src_selector = Data[offset++];
            uint8_t dst_selector = Data[offset++];
            
            std::vector<std::string> test_files = {
                "src_file.txt", "dst_file.txt", "nonexistent.txt", "./tmp_src.txt", "./tmp_dst.txt"
            };
            
            std::string src = test_files[src_selector % test_files.size()];
            std::string dst = test_files[dst_selector % test_files.size()];
            
            // Create source file
            if (src_selector % 2 == 0) {
                std::ofstream file(src);
                file << "test";
                file.close();
            }
            
            try {
                std::filesystem::rename(src, dst);
            } catch (const std::exception& e) {
                // Source might not exist, which is fine
            }
        }
        
        // Test rmdir
        if (offset + 1 < Size) {
            uint8_t dir_selector = Data[offset++];
            std::vector<std::string> test_dirs = {
                "test_rmdir", "nonexistent_dir", "./tmp_rmdir"
            };
            
            std::string dir_name = test_dirs[dir_selector % test_dirs.size()];
            
            // Create directory first to potentially remove
            if (dir_selector % 2 == 0) {
                try {
                    std::filesystem::create_directories(dir_name);
                } catch (...) {}
            }
            
            try {
                std::filesystem::remove(dir_name);
            } catch (const std::exception& e) {
                // Directory might not exist or not be empty, which is fine
            }
        }
        
        // Test listdir
        if (offset + 1 < Size) {
            uint8_t dir_selector = Data[offset++];
            std::vector<std::string> test_dirs = {
                ".", "..", "/tmp", "nonexistent_dir"
            };
            
            std::string dir_name = test_dirs[dir_selector % test_dirs.size()];
            
            try {
                for (const auto& entry : std::filesystem::directory_iterator(dir_name)) {
                    // Just iterate to test functionality
                }
            } catch (const std::exception& e) {
                // Directory might not exist, which is fine
            }
        }
        
        // Test path exists
        if (offset + 1 < Size) {
            uint8_t path_selector = Data[offset++];
            std::vector<std::string> test_paths = {
                ".", "/", "/tmp", "nonexistent_path", "/dev/null"
            };
            
            std::string path = test_paths[path_selector % test_paths.size()];
            bool exists = std::filesystem::exists(path);
        }
        
        // Test is_directory
        if (offset + 1 < Size) {
            uint8_t path_selector = Data[offset++];
            std::vector<std::string> test_paths = {
                ".", "/", "/tmp", "nonexistent_path", "/dev/null"
            };
            
            std::string path = test_paths[path_selector % test_paths.size()];
            bool is_dir = std::filesystem::is_directory(path);
        }
        
        // Test is_regular_file
        if (offset + 1 < Size) {
            uint8_t path_selector = Data[offset++];
            std::vector<std::string> test_paths = {
                "fuzzer_utils.h", "/etc/passwd", "nonexistent_file.txt", "/dev/null"
            };
            
            std::string path = test_paths[path_selector % test_paths.size()];
            bool is_file = std::filesystem::is_regular_file(path);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
