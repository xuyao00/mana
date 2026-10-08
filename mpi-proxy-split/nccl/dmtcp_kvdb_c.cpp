#include "dmtcp_kvdb_c.h"
#include "base64.h"
#include "dmtcp.h"
#include "kvdb.h"
#include <new>
#include <stdexcept>
#include <string.h>

// Distributed key-value store using DMTCP Coordinator's Name Service.
// This replaces the local in-process map to support multi-node synchronization.
//
// NOTE: the coordinator wipes this database when the first restarting worker
// connects, so nothing published before a checkpoint is readable after a
// restart. Durable state has to live in ordinary process memory (which DMTCP
// checkpoints for free). These calls are only for within-one-event rendezvous.

using namespace dmtcp::kvdb;

// Database id, capped at 64 chars by the wire protocol.
static const char *const NCCL_MANA_DB = "nccl-mana";

extern "C" int dmtcp_kvdb_incr(const char *key, int64_t val) {
    if (!key)
        return 0;
    dmtcp_disable_ckpt();
    KVDBResponse res = request64(KVDBRequest::INCRBY, NCCL_MANA_DB, key, val);
    dmtcp_enable_ckpt();
    return (res == KVDBResponse::SUCCESS);
}

extern "C" int dmtcp_kvdb_set(const char *key, int64_t val) {
    if (!key)
        return 0;
    dmtcp_disable_ckpt();
    KVDBResponse res = set64(NCCL_MANA_DB, key, val);
    dmtcp_enable_ckpt();
    return (res == KVDBResponse::SUCCESS);
}

extern "C" int64_t dmtcp_kvdb_get(const char *key) {
    if (!key)
        return 0;
    dmtcp_disable_ckpt();
    int64_t oldVal = 0;
    KVDBResponse res = get64(NCCL_MANA_DB, key, &oldVal);
    dmtcp_enable_ckpt();
    if (res == KVDBResponse::SUCCESS) {
        return oldVal;
    }
    return 0;
}

extern "C" int dmtcp_kvdb_max(const char *key, int64_t val) {
    if (!key)
        return 0;
    dmtcp_disable_ckpt();
    KVDBResponse res = request64(KVDBRequest::MAX, NCCL_MANA_DB, key, val);
    dmtcp_enable_ckpt();
    return (res == KVDBResponse::SUCCESS);
}

extern "C" int dmtcp_kvdb_fetch_add(const char *key, int64_t delta, int64_t *oldVal) {
    if (!key)
        return 0;
    dmtcp_disable_ckpt();
    int64_t prev = 0;
    KVDBResponse res = request64(KVDBRequest::INCRBY, NCCL_MANA_DB, key, delta, &prev);
    dmtcp_enable_ckpt();
    if (res != KVDBResponse::SUCCESS)
        return 0;
    if (oldVal)
        *oldVal = prev;
    return 1;
}

extern "C" int dmtcp_kvdb_set_blob(const char *key, const void *buf, size_t len) {
    if (!key || !buf || len == 0)
        return 0;

    int ok = 0;
    dmtcp_disable_ckpt();
    try {
        // Values cross the wire as NUL-terminated strings, so raw bytes would be
        // truncated at the first zero. base64 keeps a 128-byte ncclUniqueId in a
        // single round trip instead of 16 int64 chunks.
        dmtcp::string encoded = dmtcp::base64::encode((char const *)buf, len);
        ok = (set(NCCL_MANA_DB, key, encoded) == KVDBResponse::SUCCESS);
    } catch (const std::exception &) {
        ok = 0;
    }
    dmtcp_enable_ckpt();
    return ok;
}

extern "C" int dmtcp_kvdb_get_blob(const char *key, void *buf, size_t len) {
    if (!key || !buf || len == 0)
        return 0;

    int ok = 0;
    dmtcp_disable_ckpt();
    try {
        dmtcp::string encoded;
        if (get(NCCL_MANA_DB, key, &encoded) == KVDBResponse::SUCCESS && !encoded.empty()) {
            // decode() throws on a non-base64 character.
            dmtcp::string decoded = dmtcp::base64::decode(encoded);
            if (decoded.size() == len) {
                memcpy(buf, decoded.data(), len);
                ok = 1;
            }
        }
    } catch (const std::exception &) {
        ok = 0;
    }
    dmtcp_enable_ckpt();
    return ok;
}
