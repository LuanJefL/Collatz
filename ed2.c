#include <stdio.h>
#include <time.h>

#define CACHE_SIZE 50000000
long long cache[CACHE_SIZE] = {0};

long long calcular_collatz(long long n){
    long long original_n = n;
    long long passos = 0;

    while(n >= CACHE_SIZE || cache[n] == 0) {
        if (n % 2 == 0){
            n = n/2;
        } else{
            n = 3 * n + 1;
        }
        passos++;
    }
    long long tamanho_total = passos + cache[n];
    if(original_n < CACHE_SIZE) {
        cache[original_n] = tamanho_total;
    }
    return tamanho_total;
}

int main(){
    cache[1] = 1;
    long long maior_sequencia = 0;
    long long melhor_seed = 0;
    long long limite_de_busca = 1000000;
    clock_t inicio, fim;

    inicio = clock();

    for(long long i = 1; i <= limite_de_busca; i++) {
        long long tamanho_atual = calcular_collatz(i);

        if(tamanho_atual > maior_sequencia){
            maior_sequencia = tamanho_atual;
            melhor_seed = i;
        }
    }
    fim = clock();
    double tempo_gasto = ((double)(fim - inicio)) / CLOCKS_PER_SEC;

    printf("Melhor seed: %lld\n", melhor_seed);
    printf("Comprimento da sequencia: %lld\n", maior_sequencia);
    printf("----------------------------------------------------------------\n");
    printf("Tempo de execução: %.3f ms\n", tempo_gasto * 1000.0);

    return 0;
}