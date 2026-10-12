#!/bin/sh
# Pack a change set for the mechanical transport of ADR-0012 (session-0013).
#
# The bytes are produced by tar/openssl here, never typed by a model -- that is the whole point of
# the transport. The output is an opaque encrypted blob plus a manifest of `git hash-object` values
# that the materialize workflow re-checks before it commits anything.
#
# Usage: scripts/pack_payload.sh <out-dir> <file> [<file> ...]
set -eu

out="${1:?usage: pack_payload.sh <out-dir> <file>...}"
shift
[ "$#" -gt 0 ] || { echo "no files given" >&2; exit 2; }

mkdir -p "$out/tree"
: > "$out/manifest-blobs.txt"

for path in "$@"; do
    [ -f "$path" ] || { echo "missing file: $path" >&2; exit 1; }
    mkdir -p "$out/tree/$(dirname "$path")"
    cp "$path" "$out/tree/$path"
    printf '%s\t%s\n' "$(git hash-object "$path")" "$path" >> "$out/manifest-blobs.txt"
done

tar -czf "$out/tree.tgz" -C "$out" tree manifest-blobs.txt
key="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
printf '%s' "$key" | openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -pass stdin \
    -in "$out/tree.tgz" -out "$out/tree.enc"
sha="$(sha256sum "$out/tree.enc" | cut -d' ' -f1)"

echo "files: $(wc -l < "$out/manifest-blobs.txt")"
echo "key: $key"
echo "sha256: $sha"
echo "payload: $out/tree.enc"
