int
sys_printpid(void)
{
cprintf(“Process ID: %d\n”, proc->pid);
return 0
}
extern int sys_printpid(void);
