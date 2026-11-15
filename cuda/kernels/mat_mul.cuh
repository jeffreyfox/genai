#pragma once

void matMul(float* A_h, float* B_h, float* C_h, int m, int n, int k, int block_size, bool tiled=false);
