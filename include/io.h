/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */
#ifndef IO_H
#define IO_H

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "assert.h"
using std::cout;
using std::endl;
using std::string;
using std::vector;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
template<class T>
void SaveKNN(const std::vector<std::vector<T>> &knns,
             const std::string &path = "output.bin") {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
    const unsigned N = knns.size();
    assert(knns.front().size() == K);
    for (unsigned i = 0; i < N; ++i) {
        auto const &knn = knns[i];
        ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(T));
    }
    ofs.close();
}



/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
template<class T>
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<T>> &data) {
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    unsigned N;  // num of points
    ifs.read((char *)&N, sizeof(unsigned));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
    std::vector<T> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(T))) {
        std::vector<T> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++) {
            row[d] = static_cast<T>(buff[d]);
        }
        data[counter++] = std::move(row);
    }
    ifs.close();
    std::cout << "Finish Reading Data" << endl;
}
#endif