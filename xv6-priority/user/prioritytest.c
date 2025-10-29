// user/prioritytest_barrier.c
#include "types.h"
#include "stat.h"
#include "user.h"
#include "pstat.h"

#define NCH 5
static int tvals[NCH] = {10,10,30,50,50};
static int pids[NCH];

static int idx_of_pid(struct pstat *ps, int pid){
  for(int i=0;i<NPROC;i++) if(ps->inuse[i] && ps->pid[i]==pid) return i;
  return -1;
}

static void burn(void){
  volatile unsigned long x=0;
  for(int outer = 0; outer < 500; outer ++){
    for (unsigned long i=0;i<2000000UL;i++) x+=i;
  }
}

int
main(void)
{
  int pipefd[2];
  if (pipe(pipefd) < 0) {
    printf(1,"pipe failed\n"); exit();
  }
  int rfd = pipefd[0], wfd = pipefd[1];

  // fork children; each waits on barrier (read 1 byte)
  for (int i=0;i<NCH;i++){
    int pid = fork();
    if (pid == 0){
      // child
      close(wfd);             // child only reads
      char go;
      if (read(rfd,&go,1)!=1){ exit(); }
      if (setpriority(tvals[i])<0){ printf(1,"setpriority fail\n"); exit(); }
      sleep(10);
      burn();
      exit();
    }
    pids[i]=pid;
  }

  // parent: close read end, then release barrier for all at once
  close(rfd);
  char go='G';
  for (int i=0;i<NCH;i++) write(wfd,&go,1);
  close(wfd);

  // sampling
  struct pstat ps_prev, ps_now;
  // warm-up
  sleep(50);
  getpinfo(&ps_prev);

  int rounds = 10;
  int dticks[NCH]; for(int i=0;i<NCH;i++) dticks[i]=0;

  // CSV header
  printf(1,"priority,delta_ticks,share_pct\n");

  for (int r=0; r<rounds; r++){
    sleep(100); // ~100 timer ticks between samples
    if (getpinfo(&ps_now)<0){ printf(1,"getpinfo fail\n"); break; }

    // compute deltas this interval
    int sum = 0;
    int d[NCH] = {0};
    for (int i=0;i<NCH;i++){
      int k_prev = idx_of_pid(&ps_prev, pids[i]);
      int k_now  = idx_of_pid(&ps_now , pids[i]);
      if (k_prev>=0 && k_now>=0){
        int delta = ps_now.ticks[k_now] - ps_prev.ticks[k_prev];
        if (delta < 0) delta = 0; // just in case
        d[i] = delta;
        sum += delta;
      }
          }

    // print CSV for this interval
    for (int i=0;i<NCH;i++){
      int pct = (sum>0)? (100*d[i]/sum):0;
      printf(1,"%d,%d,%d\n", tvals[i], d[i], pct);
      dticks[i] += d[i];
    }

    // roll prev forward
    ps_prev = ps_now;
  }

  // final aggregate shares
  int tsum = 0; for (int i=0;i<NCH;i++) tsum += dticks[i];
  printf(1,"final shares (aggregate):\n");
  for (int i=0;i<NCH;i++){
    int pct = (tsum>0)? (100*dticks[i]/tsum):0;
    printf(1,"  priority=%d  ticks=%d  ~%d%%\n", tvals[i], dticks[i], pct);
  }

  // cleanup
  for (int i=0;i<NCH;i++) kill(pids[i]);
  for (int i=0;i<NCH;i++) wait();
  exit();
}

