#ifndef TIEMU_INTERNAL_H
#define TIEMU_INTERNAL_H

#include "tiemu_core.h"
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

// C++17 inline variables provide one TLS error slot across all translation
// units in the shared core. No Python or binding-runtime state is involved.
namespace tiemu_detail {

inline thread_local std::string error;
inline thread_local const char *emergency_error = nullptr;

inline const char *last_error() noexcept {
    return emergency_error ? emergency_error : error.c_str();
}

inline void set_error(const char *message) noexcept {
    try {
        error = message ? message : "unknown native error";
        emergency_error = nullptr;
    } catch (...) {
        // Even allocation failure while storing an exception must not escape
        // through a C entry point. This literal needs no allocation or cleanup.
        error.clear();
        emergency_error = "native error; unable to allocate error message";
    }
}

template <class Function>
int guarded(Function &&function) noexcept {
    error.clear();
    emergency_error = nullptr;
    try {
        std::forward<Function>(function)();
        return TIEMU_OK;
    } catch (const std::invalid_argument &exception) {
        set_error(exception.what());
        return TIEMU_VALUE_ERROR;
    } catch (const std::out_of_range &exception) {
        set_error(exception.what());
        return TIEMU_INDEX_ERROR;
    } catch (const std::exception &exception) {
        set_error(exception.what());
        return TIEMU_RUNTIME_ERROR;
    } catch (...) {
        set_error("unknown native exception");
        return TIEMU_RUNTIME_ERROR;
    }
}

} // namespace tiemu_detail

#endif // TIEMU_INTERNAL_H
