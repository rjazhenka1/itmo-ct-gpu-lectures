#define TILE 8

kernel void mmul(global const float *ga, global const float *gb, global float *c, const uint N, const uint K, const uint M) {
    uint x = get_global_id(0); 
    uint y = get_global_id(1);

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    local float la[TILE][TILE];
    local float lb[TILE][TILE];
    float sum = 0;
    for (uint k = 0; k < K; k += TILE) {
        if (k + lx < K && k + ly < K) {
            la[lx][ly] = ga[y * K + k + lx];
            lb[lx][ly] = gb[(k + ly) * N + x];
        }
        barrier(CLK_LOCAL_MEM_FENCE); 
        /*  
            Нам нужно убедиться, что все треды
            внутри локальной группы закончат
            все доступы к памяти до барьера,
            прежде чем начнут ходить в память
            после барьера. Барьеры есть типов
            CLK_LOCAL_MEM_FENCE и CLK_GLOBAL_MEM_FENCE.
            БАРЬЕРЫ СИНХРОНИЗИРУЮТ ТОЛЬКО ТРЕДЫ
            ВНУТРИ ОДНОЙ ЛОКАЛЬНОЙ ГРУППЫ. МЕЖДУ
            ЛОКАЛЬНЫМИ ГРУППАМИ СИНХРОНИЗАЦИИ НЕТ.
        */ 
        for (uint z = 0; z < TILE; z++) {
            sum += la[z][ly] * lb[lx][z];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * N + x] = sum;
}