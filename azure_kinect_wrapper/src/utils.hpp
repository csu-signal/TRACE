#ifndef UTILS_HPP
#define UTILS_HPP

#define K4A_VERIFY(result, error)                                              \
  if (result != K4A_RESULT_SUCCEEDED) {                                        \
    printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__,    \
           __FUNCTION__, __LINE__);                                            \
    exit(1);                                                                   \
  }

#endif
