#pragma once
#include <string>
using std::string;

const string small_data = "/dataset/sigmod2024/small/dummy-data.bin",
        small_queries = "/dataset/sigmod2024/small/dummy-queries.bin";
const string medium_data = "/dataset/sigmod2024/medium/contest-data-release-1m.bin",
        medium_queries = "/dataset/sigmod2024/medium/contest-queries-release-1m.bin",
        medium_gt = "/dataset/sigmod2024/medium/contest-gt-release-1m.bin";
const string large_data = "/dataset/sigmod2024/large/contest-data-release-10m.bin",
        large_queries = "/dataset/sigmod2024/large/contest-queries-release-10m.bin",
        large_gt = "/dataset/sigmod2024/large/contest-gt-release-10m.bin";

const string
        large4_queries = "/dataset/sigmod2024/large/fix-Public-4M-queries.bin",
        large4_gt = "/dataset/sigmod2024/large/05_contest-gt-release-10m.bin";

const string dummy_data = "dummy-data.bin";
const string dummy_queries = "dummy-queries.bin";


const string index_file = "hnsw_index_";

const int orderByT = 10023456;
const int orderByC = 1222333;
