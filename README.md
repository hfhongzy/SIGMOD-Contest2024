# SIGMODContest2024

## Team Info

Team: biejuanle (Southern University of Science and Technology and Zhejiang University)

| Name | Email | Institution |
|:--:|:--:|:--:|
| Zhaoyang Hong | hongzy2022@mail.sustech.edu.cn | Southern University of Science and Technology |
| Zhaohang Feng | 12210722@mail.sustech.edu.cn | Southern University of Science and Technology |
| Wanting Li | 12212760@mail.sustech.edu.cn | Southern University of Science and Technology |
| Jiale Zhang | 12112527@mail.sustech.edu.cn | Southern University of Science and Technology |
| Hao Wu | 2964491219@qq.com | Zhejiang University |
| Yujie Wang | wangyujie.mail@gmail.com | Southern University of Science and Technology |
| Peiran Liang | 12210726@mail.sustech.edu.cn | Southern University of Science and Technology |

## Prerequisites

Before getting started, make sure you have the following tools installed:

- CMake: >= 3.10
- G++: >= 9.4.0
- OpenMP: >= 4.0

## Compile and Run


```bash
mkdir build && cd build
cmake ..
make
./main
```

## Notes

For local testing:
1. Create your data and query files.
2. Replace the value ```dummy_data``` and ```dummy_queries``` in ```common.h``` with your dataset path.

Explanation:
1. The speed of building index and searching maybe unstable, in the competition, we submitted the same 
program repeatedly to get the best performance.

## References

Some source code is adapted from https://github.com/nmslib/hnswlib and https://github.com/zilliztech/pyglass.