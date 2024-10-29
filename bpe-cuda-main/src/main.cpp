#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdlib>


std::pair<std::string, std::string> bpe_main(std::string input_text, bool parallel);

std::mutex ioMutex;
struct FileStats {
    size_t totalChunks = 0;
    size_t totalChunkSize = 0;

    std::chrono::duration<double> totalProcessingTime;
};


void write_metrics_to_file(std::string metrics, std::string output_file_name) {
    std::ofstream outputFile(output_file_name, std::ios::app);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file - " << output_file_name << std::endl;
        return;
    }

    outputFile << "\n{" << metrics << " },";

    outputFile.close();
}

// Define a function to process a single file with CUDA.
void processFileWithCuda(const std::string &inputFile, const std::string &outputFile, bool parallel) {
    // Initialize file processing statistics.
    FileStats stats;
    const size_t chunkSize = 1024 * 1024; //1MB
    std::ifstream inFile(inputFile, std::ios::binary);

    if (!inFile) {
        std::cerr << "Could not open the input file - '" << inputFile << "'" << std::endl;
        return;
    }

    std::string chunk(chunkSize, '\0');

    // Determine the file size
    inFile.seekg(0, std::ios::end);
    size_t fileSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    int chunk_num = 0;
    int totalBytesRead = 0;
    // Loop until the end of the file is reached.
    while (totalBytesRead < fileSize) {
        size_t bytesToRead = std::min(chunkSize, fileSize - totalBytesRead);
        std::vector<char> buffer(bytesToRead);

        inFile.read(buffer.data(), bytesToRead);
        size_t bytesRead = inFile.gcount();

        if (bytesRead == 0) break;  // If no bytes were read, break out of the loop.
        totalBytesRead += bytesRead;

        auto start = std::chrono::high_resolution_clock::now();
        auto res = bpe_main(std::string(buffer.begin(), buffer.end()), parallel);

        write_metrics_to_file(res.second, "data/output/results.json");

        auto end = std::chrono::high_resolution_clock::now();
        stats.totalProcessingTime += end - start;

        std::string ofname = outputFile + "_chunk_" + std::to_string(chunk_num);
        std::ofstream outFile(ofname,
                              std::ios::binary | std::ios::out);
        outFile << res.first;
        outFile.close();

        stats.totalChunks++;
        stats.totalChunkSize += bytesRead;
    }

    system(("cat " + outputFile + "_chunk_* > " + outputFile).c_str());
    system(("rm " + outputFile + "_chunk_* ").c_str());



    // Output the file-specific processing statistics to the console.
    std::lock_guard<std::mutex> lock(ioMutex);
    std::cout << "File: " << inputFile << "\n";
    std::cout << "Total chunks processed: " << stats.totalChunks << "\n";
    std::cout << "Total processing time: " << stats.totalProcessingTime.count() << " seconds\n";
}

// Define a function to process multiple files concurrently.
void processMultipleFiles(const std::vector<std::string> &inputFiles, const std::vector<std::string> &outputFiles,
                          bool parallel) {
    // Create a thread for each file to be processed with CUDA.
    std::vector<std::thread> fileProcessingThreads;
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        fileProcessingThreads.emplace_back(processFileWithCuda, inputFiles[i], outputFiles[i], parallel);
    }

    // Wait for all threads to finish processing.
    for (auto &t: fileProcessingThreads) {
        t.join();
    }
}

// The main entry point of the program.
int main() {
    // List of input and output files to be processed.
    bool parallel = true;
    std::string input_dir = "data/input/";
    std::string output_dir = "data/output/";
    std::vector<std::string> inputFiles = {input_dir+"input1.txt",
                                           input_dir+"input2.txt",
                                           input_dir+"input3.txt",
                                           input_dir+"friends_transcripts.tsv",
                                           input_dir+"JEOPARDY_QUESTIONS1.json"
    };

    std::vector<std::string> outputFiles = {output_dir+"output1.txt",
                                            output_dir+"output2.txt",
                                            output_dir+"output3.txt",
                                            input_dir+"friends_transcripts-tokenized.txt",
                                            input_dir+"JEOPARDY_QUESTIONS1_json-tokenized.txt"
    };

//    bpe_main("hello! manu is my name", parallel);
    // Call the function to process all files concurrently.
    processMultipleFiles(inputFiles, outputFiles, parallel);

    return 0;  // Return zero to indicate successful execution.
}