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

| Length | Avg attempts |
|--------|-------------|
| 1 | ~29 |
| 2 | ~1,700 |
| 3 | ~97K |
| 4 | ~5.6M |
| 5 | ~324M |
| 6 | ~18.8B |
| 7 | ~1.1T |
| 8 | ~63T |

> Invalid characters (0, O, I, l) are not in base58 — the tool will warn you if your prefix contains them.

## Support :)
(SOL) 4ZAsLqygvpoqgmzmH54pwEipU5QZdxSkXXLnmBDGqVka

## License

MIT
