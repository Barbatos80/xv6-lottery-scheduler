#include "types.h" 
#include "stat.h" 
#include "user.h"
#include "fs.h"
int
main(void)
{
  //read global counter
  int before = firstpart();
  printf(1, "firstpart before: %d\n", before);
  //trigger a couple of getpid() calls
  getpid();
  getpid();
  int after = firstpart();
  printf(1, "firstpart after: %d\n", after);
  //fork so child also bumps the counter with getpid()
  int pid = fork();
  if(pid < 0){
    printf(1, "fork failed\n");
    exit();
  }
  if(pid == 0){
    //child
    getpid();
    printf(1, "child firstpart: %d\n", firstpart());
    exit();
  } else{
     //parent
     wait();
     printf(1, "parent firstpart: %d\n", firstpart()); 
     exit();
  }
}
