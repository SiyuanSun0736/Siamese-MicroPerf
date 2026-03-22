CC     := gcc
CFLAGS := -O2 -Wall -Wextra -Isrc
LDFLAGS := -lm

SRCDIR  := src
SRCS    := $(SRCDIR)/main.c \
           $(SRCDIR)/pmu_counters.c \
           $(SRCDIR)/lbr.c \
           $(SRCDIR)/output.c \
           $(SRCDIR)/tid_monitor.c
OBJS    := $(SRCS:.c=.o)
TARGET  := pmu_monitor

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	sudo ./$(TARGET) -i 500

test:
	$(MAKE) -C test all

clean:
	rm -f $(TARGET) $(OBJS)

