#!/system/bin/sh
# HWASan needs the platform hwasan libc (/system/lib64/hwasan), selected by
# the linker when LD_HWASAN is set. Loading the runtime as a plain DT_NEEDED
# library instead initializes it after scudo, and the tagged pointers then
# crash the allocator during __hwasan_init.
LD_HWASAN=1 exec "$@"
