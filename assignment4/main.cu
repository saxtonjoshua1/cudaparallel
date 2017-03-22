#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "timer.h"

void fatal_error (const char* message);
// precondition: message is not NULL
// postcondition: message has been written to standard error & program
//                terminated

char* create_error_string (const char* message, const char* data_string);

  
int main (int argc, char** argv)
{
  if (argc != 2)
    fatal_error("assn2 <num_intervals> <file\n");
  
  int num_intervals= (int) strtol(argv[1], (char **)NULL, 10);

  int min = 0;
  int max = 1; 

  return 0;
}

void fatal_error (const char* message)
{
  fprintf (stderr, "%s", message);
  exit (0);
}

char* create_error_string (const char* message, const char* data_string)
{
  char* result = (char*)malloc (60 * sizeof(char));
  if (result == NULL)
    fatal_error ("malloc error");
  snprintf (result, 60, message, data_string);
  return result;
}

