#include "types.h"
#include "x86.h"
#include "defs.h"
#include "param.h"
#include "mmu.h"       // for struct segdesc used by proc.h
#include "proc.h"      // struct proc, NPROC, global 'proc'
#include "spinlock.h"  // ptable
#include "pstat.h"     // struct pstat

extern struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;

int
sys_settickets(void)
{
  int n;
  if (argint(0, &n) < 0) return -1;
  if (n < 1) return -1;

  acquire(&ptable.lock);
  if (proc) {
    proc->tickets = n;     // use global 'proc' instead of myproc()
  }
  release(&ptable.lock);
  return 0;
}

int
sys_getpinfo(void)
{
  struct pstat *ps;
  // your argptr signature is (int, char **, int)
  if (argptr(0, (char **)&ps, sizeof(*ps)) < 0) return -1;

  acquire(&ptable.lock);
  int i = 0;
  for (struct proc *p = ptable.proc; p < &ptable.proc[NPROC]; p++, i++) {
    if (p->state == UNUSED) {
      ps->inuse[i]   = 0;
      ps->tickets[i] = 0;
      ps->pid[i]     = 0;
      ps->ticks[i]   = 0;
    } else {
      ps->inuse[i]   = 1;
      ps->tickets[i] = p->tickets;
      ps->pid[i]     = p->pid;
      ps->ticks[i]   = p->ticks;
    }
  }
  release(&ptable.lock);
  return 0;
}

int
sys_fork(void)
{
  return fork();
}

int
sys_exit(void)
{
  exit();
  return 0;  // not reached
}

int
sys_wait(void)
{
  return wait();
}

int
sys_kill(void)
{
  int pid;
  if (argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}

int
sys_getpid(void)
{
  return proc->pid;
}

int
sys_sbrk(void)
{
  int addr;
  int n;

  if (argint(0, &n) < 0)
    return -1;
  addr = proc->sz;
  if (growproc(n) < 0)
    return -1;
  return addr;
}

int
sys_sleep(void)
{
  int n;
  uint ticks0;

  if (argint(0, &n) < 0)
    return -1;
  acquire(&tickslock);
  ticks0 = ticks;
  while (ticks - ticks0 < (uint)n) {
    if (proc->killed) {
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

// return how many clock tick interrupts have occurred since boot.
int
sys_uptime(void)
{
  uint xticks;
  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}
