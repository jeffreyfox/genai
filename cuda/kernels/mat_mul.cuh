#pragma once

void matMul(float* A_h, float* B_h, float* C_h, int width, int n_threads, int warmup, int iters, bool tiled=false, int tile_width=32);
