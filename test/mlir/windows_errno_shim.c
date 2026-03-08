__declspec(dllimport) int *__cdecl _errno(void);

int *__errno_location(void) {
  return _errno();
}

int printf(const char *format, ...) {
  (void)format;
  return 0;
}
