#include <stdio.h>
#include <stdint.h>

#define CACHE_SIZE 10000000

// uint16_t economiza MUITA memória
// suporta até 65535 passos
static uint16_t cache[CACHE_SIZE];

uint32_t collatz(uint64_t n) {

    uint64_t path[1024];
    uint32_t depth = 0;

    while (n != 1) {

        // usa cache se disponível
        if (n < CACHE_SIZE && cache[n]) {
            depth += cache[n];
            break;
        }

        // guarda caminho temporário
        if (depth < 1024)
            path[depth] = n;

        depth++;

        // versão otimizada
        if (n & 1)
            n = (3 * n + 1) >> 1;
        else
            n >>= 1;
    }

    // salva resultados retroativamente
    uint32_t total = depth;

    for (uint32_t i = 0; i < depth && i < 1024; i++) {

        uint64_t v = path[i];

        if (v < CACHE_SIZE) {

            uint32_t steps = total - i;

            if (steps < 65535)
                cache[v] = steps;
        }
    }

    return total;
}

int main() {

    cache[1] = 1;

    uint64_t max_n = 0;
    uint32_t max_steps = 0;

    for (uint64_t i = 1; i <= 10000000ULL; i++) {

        uint32_t s = collatz(i);

        if (s > max_steps) {
            max_steps = s;
            max_n = i;
        }
    }

    printf("Maior sequencia:\n");
    printf("n = %llu\n", max_n);
    printf("passos = %u\n", max_steps);

    return 0;
}
