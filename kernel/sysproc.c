#include "types.h"
#include "x86.h"
#include "defs.h"
#include "param.h"
#include "mmu.h"
#include "proc.h"
#include "sysfunc.h"
#include "spinlock.h"
#include "pstat.h"
int partAcount = 0;
struct spinlock partAcount_lock;

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

  if(argint(0, &pid) < 0)
    return -1;
  return kill(pid);
}

//First Part of Project!!!
int
sys_firstpart(void){
  int count;
  acquire(&partAcount_lock);
  count = partAcount;
  release(&partAcount_lock);
  return count;
}

//Second Part of Project!!!
int
sys_PartB(void){
	return counterB;
}

//Third Part of Project!!!
int
sys_ThirdPart (void){
	return counterC;
}

int
sys_getpid(void)
{
  acquire(&partAcount_lock);
  partAcount++;
  release(&partAcount_lock);
  return proc->pid;
}

int
sys_sbrk(void)
{
  int addr;
  int n;

  if(argint(0, &n) < 0)
    return -1;
  addr = proc->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

int
sys_sleep(void)
{
  int n;
  uint ticks0;
  
  if(argint(0, &n) < 0)
    return -1;
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(proc->killed){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

// return how many clock tick interrupts have occurred
// since boot.
int
sys_uptime(void)
{
  uint xticks;
  
  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

int
sys_getpinfo(void)
{
    struct pstat *ptable;

    unsigned int number = (unsigned int)&ptable;

    return number;
}

int
sys_settickets(void)
{
	int mytickets;
	if(argint(0, &mytickets) < 0)
		return -1;

	else {
		return settickets(mytickets);
	}

	return 0;
}
