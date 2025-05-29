#include "../include/tsmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

// 使用MKL库的TSMM: C = A^T * B
void tsmm_mkl(const double* A, const double* B, double* C, 
             size_t m, size_t n, size_t k, MatrixLayout layout) {
    // 使用MKL的DGEMM函数计算矩阵乘法
    // C = alpha * op(A) * op(B) + beta * C
    // 其中op(A) = A^T, op(B) = B
    
    double alpha = 1.0;
    double beta = 0.0;
    
    if (layout == ROW_MAJOR) {
        // 行主序
        // 参数说明:
        // CblasRowMajor: 行主序存储
        // CblasTrans: A需要转置
        // CblasNoTrans: B不需要转置
        // m: A^T的行数
        // n: B的列数
        // k: A^T的列数/B的行数
        // alpha: 乘法系数
        // A: 矩阵A
        // lda: A的leading dimension
        // B: 矩阵B
        // ldb: B的leading dimension
        // beta: C的系数
        // C: 结果矩阵C
        // ldc: C的leading dimension
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                   m, n, k, alpha, A, m, B, n, beta, C, n);
    } else { // COL_MAJOR
        // 列主序
        // 对于列主序，需要调整参数
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
                   m, n, k, alpha, A, k, B, k, beta, C, m);
    }
} 