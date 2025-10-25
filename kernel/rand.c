#define SCHRAND_MAX ((1U << 31) - 1)
#define SCHRAND_MULT 214013
#define SCHRAND_CONST 2531011

int rseed = 707606505;

int rand(void)
{
return rseed = (rseed * SCHRAND_MULT + SCHRAND_CONST) % SCHRAND_MAX
}

