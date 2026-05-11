/*
Compile:

gcc Collatz.c -O3 -march=native -fopenmp -lpthread \
-lGLEW -lglut -lGL -o collatz

=========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <omp.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#define BLOCK_SIZE   1048576u
#define CACHE_SIZE   1000000ULL
#define TOP_GPU      32
#define RAND_CPU     32

volatile int running = 1;

volatile uint64_t tested_gpu = 0;
volatile uint64_t tested_cpu = 0;
volatile uint64_t best_seed = 1;
volatile uint64_t best_steps = 0;
volatile uint64_t current_base = 1;

static uint32_t *cache;

/* ===================================================== */
void stop_handler(int sig)
{
    running = 0;
}

/* =======================================================
COLLATZ REAL 64-BIT
======================================================= */
static inline uint64_t collatz64(uint64_t seed)
{
    uint64_t n = seed;
    uint64_t steps = 0;

    while (n != 1)
    {
        if (n < CACHE_SIZE && cache[n] && n != seed)
        {
            steps += cache[n];
            break;
        }

        if ((n & 1ULL) == 0ULL)
        {
            int tz = __builtin_ctzll(n);
            n >>= tz;
            steps += tz;
        }
        else
        {
            n = 3ULL * n + 1ULL;
            steps++;
        }
    }

    if (seed < CACHE_SIZE)
        cache[seed] = (uint32_t)steps;

    return steps;
}

/* =======================================================
UPDATE BEST REAL
======================================================= */
void update_best(uint64_t seed, uint64_t steps)
{
    #pragma omp critical
    {
        if (steps > best_steps)
        {
            best_steps = steps;
            best_seed = seed;

            printf(
                "\n[REAL RECORD] Seed=%" PRIu64
                " Steps=%" PRIu64 "\n",
                best_seed,
                best_steps
            );

            fflush(stdout);
        }
    }
}

/* =======================================================
MONITOR
======================================================= */
void *monitor_thread(void *arg)
{
    time_t start = time(NULL);

    while (running)
    {
        sleep(10);

        double sec = difftime(time(NULL), start);
        double rate =
            sec > 0 ? (tested_gpu + tested_cpu) / sec : 0;

        printf(
            "[STATUS] Base=%" PRIu64
            " | GPU=%" PRIu64
            " | CPU=%" PRIu64
            " | Best=%" PRIu64
            " (%" PRIu64 " steps)"
            " | %.0f nums/s\n",

            current_base,
            tested_gpu,
            tested_cpu,
            best_seed,
            best_steps,
            rate
        );

        fflush(stdout);
    }

    return NULL;
}

/* =======================================================
GPU SHADER
======================================================= */
const char *shader_src =
"#version 430\n"
"layout(local_size_x=256) in;\n"
"layout(std430,binding=0) buffer Out { uint data[]; };\n"
"uniform uint base;\n"
"uint fastc(uint n){\n"
" uint s=0u;\n"
" while(n>1u && s<2000u){\n"
"   if((n&1u)==0u) n>>=1u;\n"
"   else n=3u*n+1u;\n"
"   s++;\n"
" }\n"
" return s;\n"
"}\n"
"void main(){\n"
" uint id=gl_GlobalInvocationID.x;\n"
" uint seed=base + id*2u + 1u;\n"
" data[id]=fastc(seed);\n"
"}\n";

GLuint prog, ssbo;

/* ===================================================== */
typedef struct {
    uint32_t score;
    uint64_t seed;
} Cand;

/* ===================================================== */
void insert_top(Cand *top, uint32_t score, uint64_t seed)
{
    if (score <= top[TOP_GPU - 1].score) return;

    top[TOP_GPU - 1].score = score;
    top[TOP_GPU - 1].seed = seed;

    for (int i = TOP_GPU - 1; i > 0; i--)
    {
        if (top[i].score > top[i - 1].score)
        {
            Cand t = top[i];
            top[i] = top[i - 1];
            top[i - 1] = t;
        }
    }
}

/* ===================================================== */
int init_gpu()
{
    if (glewInit() != GLEW_OK) return 0;

    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &shader_src, NULL);
    glCompileShader(sh);

    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) return 0;

    prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);

    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) return 0;

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 BLOCK_SIZE * sizeof(uint32_t),
                 NULL,
                 GL_DYNAMIC_READ);

    return 1;
}

/* =======================================================
GPU WORKER
======================================================= */
void gpu_scan(uint64_t base)
{
    Cand top[TOP_GPU];

    for (int i = 0; i < TOP_GPU; i++)
    {
        top[i].score = 0;
        top[i].seed = 0;
    }

    glUseProgram(prog);

    GLuint loc = glGetUniformLocation(prog, "base");
    glUniform1ui(loc, (GLuint)base);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    glDispatchCompute(BLOCK_SIZE / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    uint32_t *ptr =
        glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    if (!ptr) return;

    for (uint32_t i = 0; i < BLOCK_SIZE; i++)
    {
        uint64_t seed = base + i * 2ULL + 1ULL;
        insert_top(top, ptr[i], seed);
    }

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    tested_gpu += BLOCK_SIZE;

    /* valida top gpu */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < TOP_GPU; i++)
    {
        if (!top[i].seed) continue;

        uint64_t real = collatz64(top[i].seed);
        update_best(top[i].seed, real);
    }

    /* valida aleatórios */
    for (int i = 0; i < RAND_CPU; i++)
    {
        uint64_t r = rand() % BLOCK_SIZE;
        uint64_t seed = base + r * 2ULL + 1ULL;

        uint64_t real = collatz64(seed);
        tested_cpu++;

        update_best(seed, real);
    }
}


void *cpu_hunter(void *arg)
{
    uint64_t n = 1;

    while (running)
    {
        uint64_t s = collatz64(n);

        tested_cpu++;
        update_best(n, s);

        n += 2;
    }

    return NULL;
}


int main(int argc, char **argv)
{
    signal(SIGINT, stop_handler);

    srand(time(NULL));

    cache = calloc(CACHE_SIZE, sizeof(uint32_t));

    pthread_t mon;
    pthread_t hunter;

    pthread_create(&mon, NULL, monitor_thread, NULL);
    pthread_create(&hunter, NULL, cpu_hunter, NULL);

    glutInit(&argc, argv);
    glutCreateWindow("Collatz");

    if (!init_gpu())
    {
        printf("GPU indisponível.\n");
        return 1;
    }

    printf("GPU + CPU hunter ativos.\n");

    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_num_procs());

    uint64_t base = 1;

    while (running)
    {
        current_base = base;
        gpu_scan(base);
        base += BLOCK_SIZE * 2ULL;
    }

    pthread_join(mon, NULL);
    pthread_join(hunter, NULL);

    printf("\n========== FINAL ==========\n");
    printf("Best seed : %" PRIu64 "\n", best_seed);
    printf("Steps     : %" PRIu64 "\n", best_steps);
    printf("GPU tested: %" PRIu64 "\n", tested_gpu);
    printf("CPU tested: %" PRIu64 "\n", tested_cpu);

    free(cache);
    return 0;
}