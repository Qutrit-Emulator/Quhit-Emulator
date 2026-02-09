# ═══════════════════════════════════════════════════════════════════════════════
# HEXSTATE ENGINE — 6-State Quantum Processor
# ═══════════════════════════════════════════════════════════════════════════════
# Build: make
# Clean: make clean
# Test:  make test

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS = -lm

TARGET  = hexstate_engine
STRESS  = stress_test
SRCS    = main.c hexstate_engine.c bigint.c
OBJS    = $(SRCS:.c=.o)

.PHONY: all clean test stress bell

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(STRESS): stress_test.o bigint.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bell_test: bell_test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Dependencies
main.o: main.c hexstate_engine.h bigint.h
hexstate_engine.o: hexstate_engine.c hexstate_engine.h bigint.h
bigint.o: bigint.c bigint.h
stress_test.o: stress_test.c hexstate_engine.h bigint.h
bell_test.o: bell_test.c hexstate_engine.h

test: $(TARGET)
	./$(TARGET) --self-test

stress: $(STRESS)
	./$(STRESS)

bell: bell_test
	./bell_test

clean:
	rm -f $(OBJS) stress_test.o bell_test.o $(TARGET) $(STRESS) bell_test
