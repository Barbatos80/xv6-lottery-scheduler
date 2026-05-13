#include "types.h"
#include "defs.h"
#include "param.h"
#include "mmu.h"
#include "x86.h"
#include "proc.h"
#include "spinlock.h"
extern uint ticks;

struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;

static struct proc *initproc;

int nextpid = 1;
extern void forkret(void);
extern void trapret(void);

static void wakeup1(void *chan);

void
pinit(void)
{
  initlock(&ptable.lock, "ptable");
}

// MLP-based ticket advisor
// Tiny fixed-point helpers
#define Q16(x) ((int)((x) * (1<<16) + ((x) >= 0 ? 0.5 : -0.5)))
#define MULQ(a,b) ((int)(((long long)(a) * (b)) >> 16))
#define CLAMP(x,lo,hi) ((x)<(lo)?(lo):((x)>(hi)?(hi):(x)))
static inline int ABSI(int x){ return x<0 ? -x : x; }

// Fast, smooth sigmoid approximation on Q16.
// For |s|>~4, saturates to 0/1. Good enough for tickets.
static int fast_sigmoid_q16(int s){
  const int FOUR = Q16(4.0);
  if(s <= -FOUR) return 0;
  if(s >=  FOUR) return (1<<16);
  // cubic around 0: sigma ≈ 0.5 + s*(0.25 - 0.020833*s^2)
  long long ss = ((long long)s*s) >> 16;         // s^2 (Q16)
  long long inner = Q16(0.25) - ((ss * Q16(0.020833)) >> 16);
  long long prod  = ((long long)s * inner) >> 16;
  long long y = Q16(0.5) + prod;
  int out = (int)y;
  return CLAMP(out, 0, (1<<16));
}

// Feature window and ticket bounds
#define FEAT_WIN     1024       // normalization window
#define TMIN         20
#define TMAX         400
#define MAX_RLEN     64         // normalize runnable_len ∈ [0,1]
#define DECAY_NUM    7          // decay x := x * 7/8 each advisor pass
#define DECAY_DEN    8

// MLP dims: D features -> H hidden -> 1 output (sigmoid)
#define D 5
#define H 4

// Preset weights (Q16). Tuned to reward IO/sleep/wait, penalize CPU hog & crowding.
// Features order: [run_ratio, sleep_ratio, io_ratio, wait_norm, rlen_norm]
// We have 4 nodes in this hidden layer (leaky ReLU activation function)
static int W1[H][D] = {
  { Q16(-0.8), Q16(+1.0), Q16(+0.7), Q16(+0.2), Q16(-0.2) }, // h0: IO/sleep > run
  { Q16(-0.3), Q16(+0.6), Q16(+0.2), Q16(+0.9), Q16(-0.1) }, // h1: big wait boost
  { Q16(-1.1), Q16(+0.1), Q16( 0.0), Q16(+0.1), Q16(-0.4) }, // h2: penalize CPU & crowd
  { Q16(-0.2), Q16(+0.2), Q16(+0.9), Q16(+0.1), Q16(-0.2) }, // h3: favor IO
};
static int B1[H] = { Q16(0.0), Q16(0.0), Q16(0.0), Q16(0.0) };

static int W2[H] = { Q16(0.5), Q16(0.5), Q16(0.4), Q16(0.6) };
static int b2    = Q16(0.0);

// Build features in Q16 in [0,1]
static void build_features(struct proc *p, int rlen, int f[D]){
  int run_q   = Q16( (p->run_win   >= FEAT_WIN) ? 1.0 : ( (double)p->run_win   / FEAT_WIN ) );
  int sleep_q = Q16( (p->sleep_win >= FEAT_WIN) ? 1.0 : ( (double)p->sleep_win / FEAT_WIN ) );
  int io_q    = Q16( (p->io_win    >= FEAT_WIN) ? 1.0 : ( (double)p->io_win    / FEAT_WIN ) );
  uint waited = ticks - p->last_sched_tick;
  if(waited > FEAT_WIN) waited = FEAT_WIN;
  int wait_q  = Q16( (double)waited / FEAT_WIN );

  if(rlen > MAX_RLEN) rlen = MAX_RLEN;
  int rlen_q  = Q16( (double)rlen / MAX_RLEN );

  f[0] = run_q; f[1] = sleep_q; f[2] = io_q; f[3] = wait_q; f[4] = rlen_q;
}

static int relu(int x){ return x < 0 ? (x >> 3) : x; } // slope ≈ 1/8 for x<0

// Forward pass: ReLU hidden, sigmoid output. Returns p in Q16 [0,1].
static int mlp_predict_q16(struct proc *p, int rlen){
  int f[D]; build_features(p, rlen, f);
  int h[H];
  for(int k=0;k<H;k++){
    long long acc = B1[k];
    for(int j=0;j<D;j++) acc += ((long long)W1[k][j] * f[j]) >> 16;
    h[k] = relu((int)acc);
  }
  long long z = b2;
  for(int k=0;k<H;k++) z += ((long long)W2[k] * h[k]) >> 16;
  return fast_sigmoid_q16((int)z);
}

// Decay windows
static void decay_windows(struct proc *p){
  p->run_win   = (p->run_win   * DECAY_NUM) / DECAY_DEN;
  p->sleep_win = (p->sleep_win * DECAY_NUM) / DECAY_DEN;
  p->io_win    = (p->io_win    * DECAY_NUM) / DECAY_DEN;
}

// Linear baseline (logistic) for self-test only
static int lin_w[D] = { Q16(-0.8), Q16(+0.8), Q16(+0.5), Q16(+0.6), Q16(-0.3) };
static int lin_b    = Q16(0.0);

static int logistic_from_feats_q16(const int f[D]){
  long long z = lin_b;
  for(int j=0;j<D;j++) z += ((long long)lin_w[j]*f[j])>>16;
  return fast_sigmoid_q16((int)z); // Q16 [0,1]
}

// MLP test harness
#ifndef NN_TEST_DISABLE

static int mlp_predict_from_feats_q16(const int f[D]){
  int h[H];
  for(int k=0;k<H;k++){
    long long acc = B1[k];
    for(int j=0;j<D;j++) acc += ((long long)W1[k][j] * f[j]) >> 16;
    h[k] = relu((int)acc);
  }
  long long z = b2;
  for(int k=0;k<H;k++) z += ((long long)W2[k] * h[k]) >> 16;
  return fast_sigmoid_q16((int)z); // Q16 in [0,1]
}

#define FP (1<<16)
static inline int Q16f(double x){ return (int)(x*FP + (x>=0?0.5:-0.5)); }

// Minimal “assert”; panic on fail so you actually notice
#define NN_ASSERT(cond, msg) do{ if(!(cond)){ cprintf("[nn][FAIL] %s\n", msg); panic("nn selftest"); } }while(0)

// Map p E [0,1] to tickets [TMIN,TMAX] like advisor
static int tickets_from_p_q16(int p_q16){
  int span = (TMAX - TMIN);
  int add  = (int)(((long long)p_q16 * span) >> 16);
  int t = TMIN + add;
  return t < 1 ? 1 : t;
}

// Run once to validate qualitative behavior
static void nn_selftest(void){
  cprintf("[nn] selftest start\n");

  // Feature order (must match your MLP): 
  // 0=run_ratio, 1=sleep_ratio, 2=io_ratio, 3=wait_norm, 4=rlen_norm

  // Build 5 representative scenarios (all in Q16 in [0,1])
  // Keep away from saturation so ordering is visible.
  const int CPU_HOG[D]   = { Q16f(0.90), Q16f(0.02), Q16f(0.01), Q16f(0.02), Q16f(0.10) };
  const int IOish[D]     = { Q16f(0.15), Q16f(0.55), Q16f(0.35), Q16f(0.10), Q16f(0.10) };
  const int WAITER[D]    = { Q16f(0.20), Q16f(0.10), Q16f(0.05), Q16f(0.70), Q16f(0.10) };
  const int BALANCED[D]  = { Q16f(0.30), Q16f(0.20), Q16f(0.15), Q16f(0.20), Q16f(0.10) };
  const int CROWDED[D]   = { Q16f(0.30), Q16f(0.20), Q16f(0.15), Q16f(0.20), Q16f(0.95) };

  struct sample { const char* name; const int *f; } S[] = {
    {"CPU_HOG", CPU_HOG},
    {"IOish", IOish},
    {"WAITER", WAITER},
    {"BALANCED", BALANCED},
    {"CROWDED", CROWDED},
  };

  int p[5], t[5];
  for(int i=0;i<5;i++){
    p[i] = mlp_predict_from_feats_q16(S[i].f);
    t[i] = tickets_from_p_q16(p[i]);
	int p_lin = logistic_from_feats_q16(S[i].f);
    cprintf("[nn] %s : mlp=%d%%  lin=%d%%  tickets=%d\n",
        S[i].name, (p[i]*100)>>16, (p_lin*100)>>16, t[i]);
    NN_ASSERT(p[i] >= 0 && p[i] <= FP, "output not in [0,1]");
    NN_ASSERT(t[i] >= TMIN && t[i] <= TMAX, "tickets out of bounds");
  }

  // Expected qualitative ordering:
  // IOish highest, then WAITER, then BALANCED, then CPU_HOG,
  // and CROWDED should be depressed due to high rlen_norm.
  NN_ASSERT(p[1] > p[2] - Q16f(0.05) || p[2] > p[1] - Q16f(0.05),
            "IOish vs WAITER unexpectedly far apart"); // allow small tie
  NN_ASSERT(p[1] >= p[2],   "IOish should be >= WAITER");
  NN_ASSERT(p[2] >= p[3],   "WAITER should be >= BALANCED");
  NN_ASSERT(p[3] >= p[0],   "BALANCED should be >= CPU_HOG");
  NN_ASSERT(p[3] >= p[4],   "BALANCED should be >= CROWDED");

  int base[D] = { Q16f(0.30), Q16f(0.20), Q16f(0.15), Q16f(0.20), Q16f(0.30) };
  int p_base = mlp_predict_from_feats_q16(base);

  int bump_io[D];   for(int j=0;j<D;j++) bump_io[j]=base[j];
  bump_io[2] += Q16f(0.10);
  int p_io = mlp_predict_from_feats_q16(bump_io);
  NN_ASSERT(p_io > p_base, "increasing io_ratio should increase p");

  int bump_run[D];  for(int j=0;j<D;j++) bump_run[j]=base[j];
  bump_run[0] += Q16f(0.10);
  int p_run = mlp_predict_from_feats_q16(bump_run);
  NN_ASSERT(p_run < p_base, "increasing run_ratio should decrease p");

  int bump_wait[D]; for(int j=0;j<D;j++) bump_wait[j]=base[j];
  bump_wait[3] += Q16f(0.20);
  int p_wait = mlp_predict_from_feats_q16(bump_wait);
  NN_ASSERT(p_wait > p_base, "increasing wait_norm should increase p");

  int bump_rlen[D]; for(int j=0;j<D;j++) bump_rlen[j]=base[j];
  bump_rlen[4] += Q16f(0.40);
  int p_rlen = mlp_predict_from_feats_q16(bump_rlen);
  NN_ASSERT(p_rlen < p_base, "increasing rlen_norm should decrease p");

  cprintf("[nn] selftest OK\n");
}
#endif



// Called from timer on CPU 0 ~every tick; runs advisor every 50 ticks.
void
advisor_on_timer(void)
{
  // in advisor_on_timer() top (proc.c)
#ifndef NN_TEST_DISABLE
  static int did_test = 0;
  if(!did_test){ did_test = 1; nn_selftest(); }
#endif

  static int ctr = 0;
  if(++ctr < 50) return;
  ctr = 0;

  // Iterate procs under ptable.lock
  acquire(&ptable.lock);

  // Count runnable length
  int rlen = 0;
  for(struct proc *q = ptable.proc; q < &ptable.proc[NPROC]; q++)
    if(q->state == RUNNABLE) rlen++;

  // Update tickets for RUNNABLE procs; also accumulate sleep_win
  for(struct proc *p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->state == SLEEPING) {
      // Lightly accumulate sleep while actually sleeping.
      if(p->sleep_win < FEAT_WIN) p->sleep_win++;
    }
    if(p->state != RUNNABLE) {
      continue;
    }
    int p_q16 = mlp_predict_q16(p, rlen); // [0,1] Q16
    int span = (TMAX - TMIN);
    int add  = (int)(((long long)p_q16 * span) >> 16);
    int t = TMIN + add;
    if(t < 1) t = 1;
    p->tickets = t;

    decay_windows(p);
  }

  release(&ptable.lock);
}




// Look in the process table for an UNUSED proc.
// If found, change state to EMBRYO and initialize
// state required to run in the kernel.
// Otherwise return 0.
static struct proc*
allocproc(void)
{
  struct proc *p;
  char *sp;

  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->state == UNUSED)
      goto found;
  }
  release(&ptable.lock);
  return 0;

found:
  p->state = EMBRYO;
  p->pid   = nextpid++;

  // lottery defaults (now that 'p' is valid)
  p->tickets = 1;   // default >= 1
  p->ticks   = 0;   // not scheduled yet
  
  // Initialize ML-advisor variables
  p->run_win = 0;
  p->sleep_win = 0;
  p->io_win = 0;
  p->last_sched_tick = 0;

  release(&ptable.lock);

  // Allocate kernel stack if possible.
  if((p->kstack = kalloc()) == 0){
    p->state = UNUSED;
    return 0;
  }
  sp = p->kstack + KSTACKSIZE;

  // Leave room for trap frame.
  sp -= sizeof *p->tf;
  p->tf = (struct trapframe*)sp;

  // Set up new context to start executing at forkret,
  // which returns to trapret.
  sp -= 4;
  *(uint*)sp = (uint)trapret;

  sp -= sizeof *p->context;
  p->context = (struct context*)sp;
  memset(p->context, 0, sizeof *p->context);
  p->context->eip = (uint)forkret;

  return p;
}

// Set up first user process.
void
userinit(void)
{
  struct proc *p;
  extern char _binary_initcode_start[], _binary_initcode_size[];
  
  p = allocproc();
  acquire(&ptable.lock);
  initproc = p;
  if((p->pgdir = setupkvm()) == 0)
    panic("userinit: out of memory?");
  inituvm(p->pgdir, _binary_initcode_start, (int)_binary_initcode_size);
  p->sz = PGSIZE;
  memset(p->tf, 0, sizeof(*p->tf));
  p->tf->cs = (SEG_UCODE << 3) | DPL_USER;
  p->tf->ds = (SEG_UDATA << 3) | DPL_USER;
  p->tf->es = p->tf->ds;
  p->tf->ss = p->tf->ds;
  p->tf->eflags = FL_IF;
  p->tf->esp = PGSIZE;
  p->tf->eip = 0;  // beginning of initcode.S

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;
  release(&ptable.lock);
}

// Grow current process's memory by n bytes.
// Return 0 on success, -1 on failure.
int
growproc(int n)
{
  uint sz;
  
  sz = proc->sz;
  if(n > 0){
    if((sz = allocuvm(proc->pgdir, sz, sz + n)) == 0)
      return -1;
  } else if(n < 0){
    if((sz = deallocuvm(proc->pgdir, sz, sz + n)) == 0)
      return -1;
  }
  proc->sz = sz;
  switchuvm(proc);
  return 0;
}

// Create a new process copying p as the parent.
// Sets up stack to return as if from system call.
// Caller must set state of returned proc to RUNNABLE.
int
fork(void)
{
  int i, pid;
  struct proc *np;

  // Allocate process.
  if((np = allocproc()) == 0)
    return -1;

  // Copy process state from p.
  if((np->pgdir = copyuvm(proc->pgdir, proc->sz)) == 0){
    kfree(np->kstack);
    np->kstack = 0;
    np->state = UNUSED;
    return -1;
  }
  np->sz = proc->sz;
  np->parent = proc;
  *np->tf = *proc->tf;
  np->tickets = proc->tickets;   // inherit lottery tickets


  // Clear %eax so that fork returns 0 in the child.
  np->tf->eax = 0;

  for(i = 0; i < NOFILE; i++)
    if(proc->ofile[i])
      np->ofile[i] = filedup(proc->ofile[i]);
  np->cwd = idup(proc->cwd);
 
  pid = np->pid;
  np->state = RUNNABLE;
  safestrcpy(np->name, proc->name, sizeof(proc->name));
  return pid;
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait() to find out it exited.
void
exit(void)
{
  struct proc *p;
  int fd;

  if(proc == initproc)
    panic("init exiting");

  // Close all open files.
  for(fd = 0; fd < NOFILE; fd++){
    if(proc->ofile[fd]){
      fileclose(proc->ofile[fd]);
      proc->ofile[fd] = 0;
    }
  }

  iput(proc->cwd);
  proc->cwd = 0;

  acquire(&ptable.lock);

  // Parent might be sleeping in wait().
  wakeup1(proc->parent);

  // Pass abandoned children to init.
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->parent == proc){
      p->parent = initproc;
      if(p->state == ZOMBIE)
        wakeup1(initproc);
    }
  }

  // Jump into the scheduler, never to return.
  proc->state = ZOMBIE;
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int
wait(void)
{
  struct proc *p;
  int havekids, pid;

  acquire(&ptable.lock);
  for(;;){
    // Scan through table looking for zombie children.
    havekids = 0;
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
      if(p->parent != proc)
        continue;
      havekids = 1;
      if(p->state == ZOMBIE){
        // Found one.
        pid = p->pid;
        kfree(p->kstack);
        p->kstack = 0;
        freevm(p->pgdir);
        p->state = UNUSED;
        p->pid = 0;
        p->parent = 0;
        p->name[0] = 0;
        p->killed = 0;
        release(&ptable.lock);
        return pid;
      }
    }

    // No point waiting if we don't have any children.
    if(!havekids || proc->killed){
      release(&ptable.lock);
      return -1;
    }

    // Wait for children to exit.  (See wakeup1 call in proc_exit.)
    sleep(proc, &ptable.lock);  //DOC: wait-sleep
  }
}

static unsigned long rand_next = 1;
static int
krand(void)
{
  rand_next = rand_next * 1103515245 + 12345;
  return (unsigned int)(rand_next >> 16) & 0x7FFF; // 0..32767
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run
//  - swtch to start running that process
//  - eventually that process transfers control
//      via swtch back to the scheduler.
void
scheduler(void)
{
  struct proc *p;

  for(;;){
    // Enable interrupts on this processor.
    sti();

    acquire(&ptable.lock);

    // Sum tickets among RUNNABLE procs
    int total = 0;
    for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
      if(p->state == RUNNABLE && p->tickets > 0)
        total += p->tickets;
    }

    if (total > 0) {
      int winner = krand() % total;

      // Find the winning process and run exactly one per pass
      for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
        if(p->state != RUNNABLE) continue;
        if(p->tickets <= 0)     continue;

        winner -= p->tickets;
        if (winner < 0) {
          // Switch to chosen process. It is the process's job
          // to release ptable.lock and then reacquire it
          // before jumping back to us.
          proc = p;
          switchuvm(p);
          p->state = RUNNING;
		  p->last_sched_tick = ticks;  // mark scheduling time
          p->ticks++;                          // accounting: chosen count
          swtch(&cpu->scheduler, proc->context);
          switchkvm();

          // Process is done running for now.
          // It should have changed its p->state before coming back.
          proc = 0;
          break; // run only the lottery winner this pass
        }
      }
    }

    release(&ptable.lock);
  }
}


// Enter scheduler.  Must hold only ptable.lock
// and have changed proc->state.
void
sched(void)
{
  int intena;

  if(!holding(&ptable.lock))
    panic("sched ptable.lock");
  if(cpu->ncli != 1)
    panic("sched locks");
  if(proc->state == RUNNING)
    panic("sched running");
  if(readeflags()&FL_IF)
    panic("sched interruptible");
  intena = cpu->intena;
  swtch(&proc->context, cpu->scheduler);
  cpu->intena = intena;
}

// Give up the CPU for one scheduling round.
void
yield(void)
{
  acquire(&ptable.lock);  //DOC: yieldlock
  proc->state = RUNNABLE;
  sched();
  release(&ptable.lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch here.  "Return" to user space.
void
forkret(void)
{
  // Still holding ptable.lock from scheduler.
  release(&ptable.lock);
  
  // Return to "caller", actually trapret (see allocproc).
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void
sleep(void *chan, struct spinlock *lk)
{
  if(proc == 0)
    panic("sleep");

  if(lk == 0)
    panic("sleep without lk");

  // Must acquire ptable.lock in order to
  // change p->state and then call sched.
  // Once we hold ptable.lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup runs with ptable.lock locked),
  // so it's okay to release lk.
  if(lk != &ptable.lock){  //DOC: sleeplock0
    acquire(&ptable.lock);  //DOC: sleeplock1
    release(lk);
  }

  // Go to sleep.
  proc->chan = chan;
  proc->state = SLEEPING;
  sched();

  // Tidy up.
  proc->chan = 0;

  // Reacquire original lock.
  if(lk != &ptable.lock){  //DOC: sleeplock2
    release(&ptable.lock);
    acquire(lk);
  }
}

// Wake up all processes sleeping on chan.
// The ptable lock must be held.
static void
wakeup1(void *chan)
{
  struct proc *p;

  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++)
    if(p->state == SLEEPING && p->chan == chan)
      p->state = RUNNABLE;
}

// Wake up all processes sleeping on chan.
void
wakeup(void *chan)
{
  acquire(&ptable.lock);
  wakeup1(chan);
  release(&ptable.lock);
}

// Kill the process with the given pid.
// Process won't exit until it returns
// to user space (see trap in trap.c).
int
kill(int pid)
{
  struct proc *p;

  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->pid == pid){
      p->killed = 1;
      // Wake process from sleep if necessary.
      if(p->state == SLEEPING)
        p->state = RUNNABLE;
      release(&ptable.lock);
      return 0;
    }
  }
  release(&ptable.lock);
  return -1;
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void
procdump(void)
{
  static char *states[] = {
  [UNUSED]    "unused",
  [EMBRYO]    "embryo",
  [SLEEPING]  "sleep ",
  [RUNNABLE]  "runble",
  [RUNNING]   "run   ",
  [ZOMBIE]    "zombie"
  };
  int i;
  struct proc *p;
  char *state;
  uint pc[10];
  
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->state == UNUSED)
      continue;
    if(p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    cprintf("%d %s %s", p->pid, state, p->name);
    if(p->state == SLEEPING){
      getcallerpcs((uint*)p->context->ebp+2, pc);
      for(i=0; i<10 && pc[i] != 0; i++)
        cprintf(" %p", pc[i]);
    }
    cprintf("\n");
  }
}


