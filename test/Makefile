CC := gcc
CFLAGS := -O2 -Wall -Wextra
LDFLAGS := -lm

SRCS := $(wildcard *.c)
TESTS := $(patsubst %.c,%, $(SRCS))

.PHONY: all clean

all: $(TESTS)

%: %.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TESTS) *.o
