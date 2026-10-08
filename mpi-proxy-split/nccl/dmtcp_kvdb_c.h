#ifndef DMTCP_KVDB_C_H
#define DMTCP_KVDB_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Increments the value at key by val. Returns 1 for success, 0 otherwise.
int dmtcp_kvdb_incr(const char *key, int64_t val);

// Sets the value at key. Returns 1 for success, 0 otherwise.
int dmtcp_kvdb_set(const char *key, int64_t val);

// Gets the value at key. Returns the value (parsed as int64_t), or 0 if not found/error.
int64_t dmtcp_kvdb_get(const char *key);

// Atomically sets key to max(current, val). A coordinator-side atomic, so every
// rank can accumulate into ONE key instead of publishing a per-rank key that
// readers then have to enumerate. On a missing key the operand is simply
// stored, so no initialization step is needed. Returns 1 for success.
int dmtcp_kvdb_max(const char *key, int64_t val);

// Atomically adds delta to key and reports the value from BEFORE the add in
// *oldVal (may be NULL). Returns 1 for success.
int dmtcp_kvdb_fetch_add(const char *key, int64_t delta, int64_t *oldVal);

// Stores an arbitrary byte buffer, base64-encoded so embedded NULs survive (the
// coordinator transports values as NUL-terminated strings). Returns 1 for
// success. len must be > 0: the coordinator rejects empty values.
int dmtcp_kvdb_set_blob(const char *key, const void *buf, size_t len);

// Reads a blob previously stored with dmtcp_kvdb_set_blob. Returns 1 only if the
// key exists and decodes to exactly len bytes; 0 otherwise, leaving buf
// untouched. A short or oversized value is a protocol error, not something to
// paper over with a partial copy.
int dmtcp_kvdb_get_blob(const char *key, void *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif // DMTCP_KVDB_C_H
