gcc -c -O2 -I/usr/include/opencv -MMD -MP -MF occam.o.d -o occam.o occam.c
gcc -o occam occam.o -lcv -lcvaux -lcxcore -lhighgui
