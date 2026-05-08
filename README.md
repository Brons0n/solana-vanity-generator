# solana-vanity

Generate custom Solana address generator written in Rust. Finds keypairs where the base58 public key starts or ends with your chosen string. Multi-threaded CPU by default, optional OpenCL GPU support.

## Build

```sh
git clone https://github.com/yourname/solana-vanity
cd solana-vanity
cargo build --release
```

With GPU (OpenCL):
```sh
cargo build --release --features gpu
```

## Run

```sh
# prefix
./target/release/solana-vanity --prefix BRON

# suffix
./target/release/solana-vanity --suffix COOL

# both
./target/release/solana-vanity --prefix BRON --suffix COOL

# case-insensitive
./target/release/solana-vanity --prefix bron --case-insensitive

# benchmark your hardware
./target/release/solana-vanity --benchmark
```

Output saves to `keypair.json` (64-byte secret+public, compatible with Solana CLI and Phantom import).

## Probability

Solana addresses are base58 (58 possible characters per position). Each character you add multiplies difficulty by 58.

| Length | Avg attempts | CPU ~5M/s | GPU ~200M/s |
|--------|-------------|-----------|-------------|
| 1 | ~29 | <1s | <1s |
| 2 | ~1,700 | <1s | <1s |
| 3 | ~97K | <1s | <1s |
| 4 | ~5.6M | ~1s | <1s |
| 5 | ~324M | ~1m | ~2s |
| 6 | ~18.8B | ~1h | ~1.5m |
| 7 | ~1.1T | ~2.5d | ~1.5h |
| 8 | ~63T | ~145d | ~4d |

> Invalid characters (0, O, I, l) are not in base58 — the tool will warn you if your prefix contains them.

## License

MIT
