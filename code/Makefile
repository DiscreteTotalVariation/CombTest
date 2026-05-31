CC     = gcc
CFLAGS = -O3 -march=native -fopenmp
LDFLAGS = -lgmp -fopenmp

# Link jemalloc if available: eliminates GMP allocator contention under many
# threads (mpz_realloc goes through per-thread arenas instead of a global lock)
# Install: sudo apt install libjemalloc-dev
JEMALLOC := $(shell pkg-config --libs jemalloc 2>/dev/null)

ct_c: ct.c
	$(CC) $(CFLAGS) -o ct_c ct.c $(LDFLAGS) $(JEMALLOC)
	@if [ -n "$(JEMALLOC)" ]; then echo "Built with jemalloc"; else echo "jemalloc not found (apt install libjemalloc-dev for faster multi-threaded runs)"; fi

ct_save: ct_save.c
	$(CC) $(CFLAGS) -o ct_save ct_save.c $(LDFLAGS) $(JEMALLOC)
	@if [ -n "$(JEMALLOC)" ]; then echo "Built with jemalloc"; else echo "jemalloc not found (apt install libjemalloc-dev for faster multi-threaded runs)"; fi

clean:
	rm -f ct_c ct_save
