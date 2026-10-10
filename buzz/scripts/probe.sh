#!/usr/bin/env bash
# Mede o que a VPS realmente tem. Cole a saída de volta para calibrarmos o modelo.
echo "== CPU"; lscpu | grep -E 'Model name|^CPU\(s\)|Flags' | sed 's/Flags:.*\(avx2\).*/Flags: ...avx2.../' ; grep -o -w -E 'avx2|avx512f|fma|neon' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "== RAM/swap"; free -m
echo "== Disco"; df -h / | tail -1
echo "== Kernel"; uname -srm
