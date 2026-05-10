use clap::Parser;
use colored::Colorize;
use crossbeam_channel::bounded;
use ed25519_dalek::Keypair;
use rand::rngs::OsRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "gpu")]
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

// ─── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about = "Solana vanity address generator")]
struct Cli {
    #[arg(long, default_value = "")]
    prefix: String,

    #[arg(long, default_value = "")]
    suffix: String,

    #[arg(long, default_value = "keypair.json")]
    output: String,

    #[arg(long)]
    case_insensitive: bool,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    gpu_only: bool,

    #[arg(long)]
    cpu_only: bool,

    #[arg(long)]
    benchmark: bool,
}

// ─── OPENCL KERNEL ──────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
const OPENCL_KERNEL: &str = r#"
// ---- xorshift128+ PRNG ----
typedef ulong2 xs_state;

void xs_init(__private xs_state *s, ulong seed) {
    s->x = seed ^ 0x9e3779b97f4a7c15UL;
    s->y = seed * 6364136223846793005UL + 1442695040888963407UL;
    if (s->x == 0) s->x = 1;
    if (s->y == 0) s->y = 1;
}

ulong xs_next(__private xs_state *s) {
    ulong t = s->x;
    ulong const c = s->y;
    s->x = c;
    t ^= t << 23;
    t ^= t >> 17;
    t ^= c ^ (c >> 26);
    s->y = t;
    return t + c;
}

void rand_bytes_32(__private xs_state *s, __private uchar *out) {
    for (int i = 0; i < 4; i++) {
        ulong v = xs_next(s);
        out[i*8+0] = (uchar)(v & 0xff);
        out[i*8+1] = (uchar)((v >> 8) & 0xff);
        out[i*8+2] = (uchar)((v >> 16) & 0xff);
        out[i*8+3] = (uchar)((v >> 24) & 0xff);
        out[i*8+4] = (uchar)((v >> 32) & 0xff);
        out[i*8+5] = (uchar)((v >> 40) & 0xff);
        out[i*8+6] = (uchar)((v >> 48) & 0xff);
        out[i*8+7] = (uchar)((v >> 56) & 0xff);
    }
}

// ---- SHA-512 ----
#define ROTR64(x,n) (((x) >> (n)) | ((x) << (64-(n))))
#define Ch(e,f,g)   (((e)&(f))^(~(e)&(g)))
#define Maj(a,b,c)  (((a)&(b))^((a)&(c))^((b)&(c)))
#define S0(a)       (ROTR64(a,28)^ROTR64(a,34)^ROTR64(a,39))
#define S1(e)       (ROTR64(e,14)^ROTR64(e,18)^ROTR64(e,41))
#define s0(x)       (ROTR64(x,1) ^ROTR64(x,8) ^((x)>>7))
#define s1(x)       (ROTR64(x,19)^ROTR64(x,61)^((x)>>6))

__constant ulong SHA512_K[80] = {
    0x428a2f98d728ae22UL,0x7137449123ef65cdUL,0xb5c0fbcfec4d3b2fUL,0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL,0x59f111f1b605d019UL,0x923f82a4af194f9bUL,0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL,0x12835b0145706fbeUL,0x243185be4ee4b28cUL,0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL,0x80deb1fe3b1696b1UL,0x9bdc06a725c71235UL,0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL,0xefbe4786384f25e3UL,0x0fc19dc68b8cd5b5UL,0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL,0x4a7484aa6ea6e483UL,0x5cb0a9dcbd41fbd4UL,0x76f988da831153b5UL,
    0x983e5152ee66dfabUL,0xa831c66d2db43210UL,0xb00327c898fb213fUL,0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL,0xd5a79147930aa725UL,0x06ca6351e003826fUL,0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL,0x2e1b21385c26c926UL,0x4d2c6dfc5ac42aedUL,0x53380d139d95b3dfUL,
    0x650a73548baf63deUL,0x766a0abb3c77b2a8UL,0x81c2c92e47edaee6UL,0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL,0xa81a664bbc423001UL,0xc24b8b70d0f89791UL,0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL,0xd69906245565a910UL,0xf40e35855771202aUL,0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL,0x1e376c085141ab53UL,0x2748774cdf8eeb99UL,0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL,0x4ed8aa4ae3418acbUL,0x5b9cca4f7763e373UL,0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL,0x78a5636f43172f60UL,0x84c87814a1f0ab72UL,0x8cc702081a6439ecUL,
    0x90befffa23631e28UL,0xa4506cebde82bde9UL,0xbef9a3f7b2c67915UL,0xc67178f2e372532bUL,
    0xca273eceea26619cUL,0xd186b8c721c0c207UL,0xeada7dd6cde0eb1eUL,0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL,0x0a637dc5a2c898a6UL,0x113f9804bef90daeUL,0x1b710b35131c471bUL,
    0x28db77f523047d84UL,0x32caab7b40c72493UL,0x3c9ebe0a15c9bebcUL,0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL,0x597f299cfc657e2aUL,0x5fcb6fab3ad6faecUL,0x6c44198c4a475817UL
};

void sha512_block(ulong *h, __private const ulong *m) {
    ulong w[80];
    for (int i = 0; i < 16; i++) w[i] = m[i];
    for (int i = 16; i < 80; i++)
        w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];

    ulong a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 80; i++) {
        ulong T1 = hh + S1(e) + Ch(e,f,g) + SHA512_K[i] + w[i];
        ulong T2 = S0(a) + Maj(a,b,c);
        hh=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
    h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// sha512 of exactly 32 bytes
void sha512_32(__private const uchar *in, __private uchar *out) {
    ulong h[8] = {
        0x6a09e667f3bcc908UL,0xbb67ae8584caa73bUL,
        0x3c6ef372fe94f82bUL,0xa54ff53a5f1d36f1UL,
        0x510e527fade682d1UL,0x9b05688c2b3e6c1fUL,
        0x1f83d9abfb41bd6bUL,0x5be0cd19137e2179UL
    };
    ulong m[16];
    for (int i = 0; i < 16; i++) m[i] = 0;
    // copy 32 bytes big-endian into m[0..3]
    for (int i = 0; i < 4; i++) {
        m[i] = ((ulong)in[i*8+0]<<56)|((ulong)in[i*8+1]<<48)|
                ((ulong)in[i*8+2]<<40)|((ulong)in[i*8+3]<<32)|
                ((ulong)in[i*8+4]<<24)|((ulong)in[i*8+5]<<16)|
                ((ulong)in[i*8+6]<<8) |((ulong)in[i*8+7]);
    }
    m[4]  = 0x8000000000000000UL;
    m[15] = 256UL; // bit length
    sha512_block(h, m);
    for (int i = 0; i < 8; i++) {
        out[i*8+0]=(uchar)(h[i]>>56); out[i*8+1]=(uchar)(h[i]>>48);
        out[i*8+2]=(uchar)(h[i]>>40); out[i*8+3]=(uchar)(h[i]>>32);
        out[i*8+4]=(uchar)(h[i]>>24); out[i*8+5]=(uchar)(h[i]>>16);
        out[i*8+6]=(uchar)(h[i]>>8);  out[i*8+7]=(uchar)(h[i]);
    }
}

// ---- Ed25519 field arithmetic (TweetNaCl-derived) ----
typedef long gf[16];

__constant long D_c[16]  = {0x78a3,0x1359,0x4dca,0x75eb,0xd8ab,0x4141,0x0a4d,0x0070,0xe898,0x7779,0x4079,0x8cc7,0xfe73,0x2b6f,0x6cee,0x5203};
__constant long D2_c[16] = {0xf159,0x26b2,0x9b94,0xebd6,0xb156,0x8283,0x149a,0x00e0,0xd130,0xeef3,0x80f2,0x198e,0xfce7,0x56df,0xd9dc,0x2406};
__constant long X_c[16]  = {0xd51a,0x8f25,0x2d60,0xc956,0xa7b2,0x9525,0xc760,0x692c,0xdc5c,0xfdd6,0xe231,0xc0a4,0x53fe,0xcd6e,0x36d3,0x2169};
__constant long Y_c[16]  = {0x6658,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666,0x6666};
__constant long I_c[16]  = {0xa0b0,0x4a0e,0x1b27,0xc4ee,0xe478,0xad2f,0x1806,0x2f43,0xd7a7,0x3dfb,0x0099,0x2b4d,0xdf0b,0x4fc1,0x2480,0x2b83};

void gf_car(__private long *o) {
    long c;
    for (int i = 0; i < 16; i++) {
        o[i] += (1L<<16);
        c = o[i] >> 16;
        o[(i+1) & 15] += c - 1 + 37*(c-1) * (i==15 ? 1 : 0);
        o[i] -= c<<16;
    }
}

void gf_add(__private long *o, __private const long *a, __private const long *b) {
    for (int i=0;i<16;i++) o[i]=a[i]+b[i];
}
void gf_sub(__private long *o, __private const long *a, __private const long *b) {
    for (int i=0;i<16;i++) o[i]=a[i]-b[i];
}
void gf_mul(__private long *o, __private const long *a, __private const long *b) {
    long t[31];
    for (int i=0;i<31;i++) t[i]=0;
    for (int i=0;i<16;i++)
        for (int j=0;j<16;j++)
            t[i+j]+=a[i]*b[j];
    for (int i=0;i<15;i++) t[i]+=38*t[i+16];
    for (int i=0;i<16;i++) o[i]=t[i];
    gf_car(o); gf_car(o);
}
void gf_sq(__private long *o, __private const long *a) { gf_mul(o,a,a); }
void gf_inv(__private long *o, __private const long *i) {
    long c[16];
    for(int j=0;j<16;j++) c[j]=i[j];
    for(int j=253;j>=0;j--) {
        gf_sq(c,c);
        if(j!=2&&j!=4) gf_mul(c,c,i);
    }
    for(int j=0;j<16;j++) o[j]=c[j];
}

void pack25519(__private uchar *o, __private long *n) {
    long m[16], t[16];
    for(int i=0;i<16;i++) t[i]=n[i];
    gf_car(t); gf_car(t); gf_car(t);
    for(int j=0;j<2;j++) {
        m[0]=t[0]-0xffed;
        for(int i=1;i<15;i++) {
            m[i]=t[i]-0xffff-((m[i-1]>>16)&1);
            m[i-1]&=0xffff;
        }
        m[15]=t[15]-0x7fff-((m[14]>>16)&1);
        int b=(int)((m[15]>>16)&1);
        m[14]&=0xffff;
        // cswap
        for(int i=0;i<16;i++) {
            long swap = (1-b)*(t[i]-m[i]);
            t[i] -= swap; m[i] += swap;
        }
    }
    for(int i=0;i<16;i++) {
        o[2*i]  =(uchar)(t[i]&0xff);
        o[2*i+1]=(uchar)(t[i]>>8);
    }
}

void unpack25519(__private long *o, __private const uchar *n) {
    for(int i=0;i<16;i++) o[i]=n[2*i]+((long)n[2*i+1]<<8);
    o[15]&=0x7fff;
}

int par25519(__private const long *a) {
    uchar d[32];
    long tmp[16];
    for(int i=0;i<16;i++) tmp[i]=a[i];
    pack25519(d,tmp);
    return d[0]&1;
}

void cswap(__private long *p, __private long *q, int b) {
    long t, c = ~((long)b - 1);
    for(int i=0;i<16;i++) { t=c&(p[i]^q[i]); p[i]^=t; q[i]^=t; }
}

// Edwards point addition
void add_point(__private long *p0,__private long *p1,__private long *p2,__private long *p3,
               __private const long *q0,__private const long *q1,__private const long *q2,__private const long *q3) {
    long a[16],b[16],c[16],d[16],e[16],f[16],g[16],h[16],t[16];
    long D[16],D2[16];
    for(int i=0;i<16;i++){D[i]=D_c[i];D2[i]=D2_c[i];}
    gf_sub(a,p1,p0); gf_sub(t,q1,q0); gf_mul(a,a,t);
    gf_add(b,p0,p1); gf_add(t,q0,q1); gf_mul(b,b,t);
    gf_mul(c,p3,q3); gf_mul(c,c,D2);
    gf_mul(d,p2,q2); gf_add(d,d,d);
    gf_sub(e,b,a); gf_sub(f,d,c); gf_add(g,d,c); gf_add(h,b,a);
    gf_mul(p0,e,f); gf_mul(p1,h,g); gf_mul(p2,g,f); gf_mul(p3,e,h);
}

void scalarmult(__private long *p0,__private long *p1,__private long *p2,__private long *p3,
                __private const uchar *s) {
    long q0[16]={0},q1[16],q2[16]={0},q3[16]={0};
    long X[16],Y[16],I[16];
    for(int i=0;i<16;i++){X[i]=X_c[i];Y[i]=Y_c[i];I[i]=I_c[i];}
    q1[0]=1; q2[0]=1;
    // copy base
    for(int i=0;i<16;i++){p0[i]=X[i];p1[i]=Y[i];p2[i]=1;} p3[0]=0;
    gf_mul(p3,X,Y);
    for(int i=0;i<16;i++){q0[i]=p0[i];q1[i]=p1[i];q2[i]=p2[i];q3[i]=p3[i];}
    // identity
    for(int i=0;i<16;i++){p0[i]=0;p1[i]=0;p2[i]=1;p3[i]=0;}
    p1[0]=1;

    for(int i=254;i>=0;i--) {
        int b=(s[i/8]>>(i&7))&1;
        cswap(p0,q0,b); cswap(p1,q1,b); cswap(p2,q2,b); cswap(p3,q3,b);
        add_point(p0,p1,p2,p3,q0,q1,q2,q3);
        add_point(q0,q1,q2,q3,p0,p1,p2,p3);
        cswap(p0,q0,b); cswap(p1,q1,b); cswap(p2,q2,b); cswap(p3,q3,b);
    }
}

void scalarbase(__private long *p0,__private long *p1,__private long *p2,__private long *p3,
                __private const uchar *s) {
    scalarmult(p0,p1,p2,p3,s);
}

void pack_point(__private uchar *r, __private long *p0,__private long *p1,__private long *p2) {
    long tx[16],ty[16],zi[16];
    gf_inv(zi,p2);
    gf_mul(tx,p0,zi);
    gf_mul(ty,p1,zi);
    pack25519(r,ty);
    r[31]^=(par25519(tx)<<7);
}

// ---- Base58 check for prefix ----
__constant uchar B58[58] = {
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
};

int base58_prefix_match(__private const uchar *pubkey, __global const uchar *prefix, int plen, int ci) {
    // encode first enough bytes of pubkey to get plen base58 chars
    // Solana pubkey is 32 bytes, base58 encodes to ~44 chars
    // We'll encode all 32 bytes
    ulong num[32];
    for(int i=0;i<32;i++) num[i]=(ulong)pubkey[i];

    uchar b58out[44];
    int outlen = 44;
    for(int i=0;i<44;i++) b58out[i]=0;

    // simple big-number base58 encode
    for(int i=0;i<32;i++){
        ulong carry=(ulong)pubkey[i];
        for(int j=43;j>=0;j--){
            carry+=256UL*(ulong)b58out[j];
            b58out[j]=(uchar)(carry%58);
            carry/=58;
        }
    }

    // find start
    int start=0;
    while(start<43 && b58out[start]==0) start++;

    for(int i=0;i<plen;i++){
        if(start+i>=44) return 0;
        uchar c=B58[b58out[start+i]];
        uchar p=prefix[i];
        if(ci){
            // lowercase both
            if(c>='A'&&c<='Z') c+=32;
            if(p>='A'&&p<='Z') p+=32;
        }
        if(c!=p) return 0;
    }
    return 1;
}

#define BATCH 512

__kernel void vanity_search(
    __global uchar *out_pubkey,
    __global uchar *out_privkey,
    __global int   *found,
    __global const uchar *prefix,
    int plen,
    __global const uchar *suffix,
    int slen,
    int ci,
    ulong seed
) {
    int gid = get_global_id(0);
    if (*found) return;

    xs_state rng;
    xs_init(&rng, (ulong)gid + seed);

    for(int iter=0; iter<BATCH; iter++){
        if(*found) return;

        uchar sk[32];
        rand_bytes_32(&rng, sk);

        uchar h[64];
        sha512_32(sk, h);

        // clamp
        h[0]  &= 248;
        h[31] &= 127;
        h[31] |= 64;

        long p0[16],p1[16],p2[16],p3[16];
        scalarbase(p0,p1,p2,p3,h);

        uchar pk[32];
        pack_point(pk,p0,p1,p2);

        int match = 1;
        if(plen>0 && !base58_prefix_match(pk,prefix,plen,ci)) match=0;
        if(slen>0 && match){
            // suffix check: encode full pubkey and check last slen chars
            uchar b58out[44];
            for(int i=0;i<44;i++) b58out[i]=0;
            for(int i=0;i<32;i++){
                ulong carry=(ulong)pk[i];
                for(int j=43;j>=0;j--){
                    carry+=256UL*(ulong)b58out[j];
                    b58out[j]=(uchar)(carry%58);
                    carry/=58;
                }
            }
            int start=0;
            while(start<43&&b58out[start]==0) start++;
            int total=44-start;
            if(total<slen){match=0;}
            else{
                for(int i=0;i<slen;i++){
                    uchar c=B58[b58out[44-slen+i]];
                    uchar s2=suffix[i];
                    if(ci){if(c>='A'&&c<='Z')c+=32;if(s2>='A'&&s2<='Z')s2+=32;}
                    if(c!=s2){match=0;break;}
                }
            }
        }

        if(match){
            int prev = atomic_xchg(found, 1);
            if(prev==0){
                for(int i=0;i<32;i++) out_pubkey[i]=pk[i];
                for(int i=0;i<32;i++) out_privkey[i]=sk[i];
                // store full 64-byte keypair: sk||pk
                for(int i=32;i<64;i++) out_privkey[i]=pk[i-32];
            }
            return;
        }
    }
}
"#;

// ─── CPU HARDWARE DETECTION ─────────────────────────────────────────────────

fn detect_cpu() -> (usize, usize, String) {
    let logical = num_cpus::get();
    let physical = num_cpus::get_physical();
    let name = {
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("model name"))
                        .and_then(|l| l.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_else(|| "Unknown CPU".to_string())
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            "Unknown CPU".to_string()
        }
    };
    (logical, physical, name)
}

fn benchmark_cpu(threads: usize, duration: Duration) -> f64 {
    let counter = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let counter = counter.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut csprng = OsRng;
                while !stop.load(Ordering::Relaxed) {
                    let _kp = Keypair::generate(&mut csprng);
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    std::thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }

    counter.load(Ordering::Relaxed) as f64 / duration.as_secs_f64()
}

// ─── GPU HARDWARE DETECTION ──────────────────────────────────────────────────

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct DeviceInfo {
    platform_idx: usize,
    device_idx: usize,
    name: String,
    vendor: String,
}

#[cfg(feature = "gpu")]
fn detect_gpu() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    let platforms = Platform::list();
    for (pi, platform) in platforms.iter().enumerate() {
        if let Ok(devs) = Device::list_all(platform) {
            for (di, dev) in devs.iter().enumerate() {
                let name = dev.name().unwrap_or_else(|_| "Unknown".to_string());
                let vendor = dev.vendor().unwrap_or_else(|_| "Unknown".to_string());
                devices.push(DeviceInfo {
                    platform_idx: pi,
                    device_idx: di,
                    name,
                    vendor,
                });
            }
        }
    }
    devices
}

#[cfg(feature = "gpu")]
fn benchmark_gpu(device: &DeviceInfo, duration: Duration) -> f64 {
    let global_size = 4096usize;
    let batch = 512u64;

    let platforms = Platform::list();
    let platform = match platforms.get(device.platform_idx) {
        Some(p) => *p,
        None => return 0.0,
    };
    let devs = match Device::list_all(&platform) {
        Ok(d) => d,
        Err(_) => return 0.0,
    };
    let dev = match devs.get(device.device_idx) {
        Some(d) => *d,
        None => return 0.0,
    };

    let ctx = match Context::builder().platform(platform).devices(dev).build() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let queue = match Queue::new(&ctx, dev, None) {
        Ok(q) => q,
        Err(_) => return 0.0,
    };
    let prog = match Program::builder()
        .src(OPENCL_KERNEL)
        .devices(dev)
        .build(&ctx)
    {
        Ok(p) => p,
        Err(_) => return 0.0,
    };

    let out_pub: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(32)
        .build()
        .unwrap();
    let out_priv: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(64)
        .build()
        .unwrap();
    let found_buf: Buffer<i32> = Buffer::builder()
        .queue(queue.clone())
        .len(1)
        .build()
        .unwrap();
    let prefix_buf: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(1)
        .build()
        .unwrap();
    let suffix_buf: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(1)
        .build()
        .unwrap();

    let kernel = match Kernel::builder()
        .program(&prog)
        .name("vanity_search")
        .queue(queue.clone())
        .global_work_size(global_size)
        .arg(&out_pub)
        .arg(&out_priv)
        .arg(&found_buf)
        .arg(&prefix_buf)
        .arg(0i32)
        .arg(&suffix_buf)
        .arg(0i32)
        .arg(0i32)
        .arg(0u64)
        .build()
    {
        Ok(k) => k,
        Err(_) => return 0.0,
    };

    let start = Instant::now();
    let mut iters = 0u64;
    while start.elapsed() < duration {
        unsafe {
            let _ = kernel.enq();
        }
        let _ = queue.finish();
        iters += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    (iters * global_size as u64 * batch) as f64 / elapsed
}

// ─── HARDWARE TABLE ──────────────────────────────────────────────────────────

fn print_hardware_table(cpu_name: &str, cpu_speed: f64, selected_gpu: Option<&str>, gpu_speed: f64) {
    println!();
    println!("{}", "┌─────────────────────────────────────────────────────────┐".cyan());
    println!("{}", "│              Hardware Performance Table                  │".cyan());
    println!("{}", "├──────────────┬──────────────────────┬────────────────────┤".cyan());
    println!("{}", "│ Device       │ Name                 │ Speed (keys/s)     │".cyan());
    println!("{}", "├──────────────┼──────────────────────┼────────────────────┤".cyan());

    let cpu_marker = if selected_gpu.is_none() { " ◄" } else { "  " };
    println!(
        "│ {:<12} │ {:<20} │ {:<18} │",
        format!("CPU{}", cpu_marker),
        &cpu_name[..cpu_name.len().min(20)],
        format!("{:.0}", cpu_speed)
    );

    if gpu_speed > 0.0 {
        let gpu_marker = if selected_gpu.is_some() { " ◄" } else { "  " };
        println!(
            "│ {:<12} │ {:<20} │ {:<18} │",
            format!("GPU{}", gpu_marker),
            selected_gpu.unwrap_or("Unknown")[..selected_gpu.unwrap_or("Unknown").len().min(20)].to_string(),
            format!("{:.0}", gpu_speed)
        );
    }

    println!("{}", "└──────────────┴──────────────────────┴────────────────────┘".cyan());
    println!();
}

// ─── CPU WORKER ──────────────────────────────────────────────────────────────

fn cpu_worker(
    prefix: &str,
    suffix: &str,
    case_insensitive: bool,
    threads: usize,
    counter: Arc<AtomicU64>,
    found: Arc<AtomicBool>,
) -> Option<(String, String, Vec<u8>)> {
    let (tx, rx) = bounded(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(|| {
            let prefix = prefix.to_string();
            let suffix = suffix.to_string();

            (0..threads).into_par_iter().for_each(|_| {
                let mut csprng = OsRng;
                loop {
                    if found.load(Ordering::Relaxed) {
                        break;
                    }
                    let kp = Keypair::generate(&mut csprng);
                    counter.fetch_add(1, Ordering::Relaxed);

                    let pubkey_bytes = kp.public.to_bytes();
                    let pubkey_b58 = bs58::encode(&pubkey_bytes).into_string();

                    let pub_cmp = if case_insensitive {
                        pubkey_b58.to_lowercase()
                    } else {
                        pubkey_b58.clone()
                    };
                    let pre_cmp = if case_insensitive {
                        prefix.to_lowercase()
                    } else {
                        prefix.clone()
                    };
                    let suf_cmp = if case_insensitive {
                        suffix.to_lowercase()
                    } else {
                        suffix.clone()
                    };

                    let matches = (prefix.is_empty() || pub_cmp.starts_with(&pre_cmp))
                        && (suffix.is_empty() || pub_cmp.ends_with(&suf_cmp));

                    if matches {
                        if !found.swap(true, Ordering::Relaxed) {
                            let secret_bytes = kp.secret.to_bytes();
                            let mut full = Vec::with_capacity(64);
                            full.extend_from_slice(&secret_bytes);
                            full.extend_from_slice(&pubkey_bytes);
                            let privkey_b58 = bs58::encode(&full).into_string();
                            let _ = tx.send((pubkey_b58, privkey_b58, full));
                        }
                        break;
                    }
                }
            });
        });

    rx.try_recv().ok()
}

// ─── GPU WORKER ──────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
fn gpu_worker(
    device: &DeviceInfo,
    prefix: &str,
    suffix: &str,
    case_insensitive: bool,
    counter: Arc<AtomicU64>,
    found_flag: Arc<AtomicBool>,
) -> Option<(String, String, Vec<u8>)> {
    let global_size = 8192usize;
    let batch_size = 512u64;

    let platforms = Platform::list();
    let platform = *platforms.get(device.platform_idx)?;
    let devs = Device::list_all(&platform).ok()?;
    let dev = *devs.get(device.device_idx)?;

    let ctx = Context::builder()
        .platform(platform)
        .devices(dev)
        .build()
        .ok()?;
    let queue = Queue::new(&ctx, dev, None).ok()?;
    let prog = Program::builder()
        .src(OPENCL_KERNEL)
        .devices(dev)
        .build(&ctx)
        .ok()?;

    let prefix_bytes = prefix.as_bytes().to_vec();
    let suffix_bytes = suffix.as_bytes().to_vec();
    let plen = prefix_bytes.len() as i32;
    let slen = suffix_bytes.len() as i32;

    let prefix_buf = Buffer::builder()
        .queue(queue.clone())
        .len(prefix_bytes.len().max(1))
        .copy_host_slice(&prefix_bytes)
        .build()
        .ok()?;
    let suffix_buf = Buffer::builder()
        .queue(queue.clone())
        .len(suffix_bytes.len().max(1))
        .copy_host_slice(&suffix_bytes)
        .build()
        .ok()?;

    let out_pub: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(32)
        .build()
        .ok()?;
    let out_priv: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .len(64)
        .build()
        .ok()?;
    let found_buf: Buffer<i32> = Buffer::builder()
        .queue(queue.clone())
        .len(1)
        .copy_host_slice(&[0i32])
        .build()
        .ok()?;

    let mut rng = OsRng;
    let mut seed_bytes = [0u8; 8];
    use rand::RngCore;
    rng.fill_bytes(&mut seed_bytes);
    let seed = u64::from_le_bytes(seed_bytes);

    let kernel = Kernel::builder()
        .program(&prog)
        .name("vanity_search")
        .queue(queue.clone())
        .global_work_size(global_size)
        .arg(&out_pub)
        .arg(&out_priv)
        .arg(&found_buf)
        .arg(&prefix_buf)
        .arg(plen)
        .arg(&suffix_buf)
        .arg(slen)
        .arg(case_insensitive as i32)
        .arg(seed)
        .build()
        .ok()?;

    let mut current_seed = seed;

    loop {
        if found_flag.load(Ordering::Relaxed) {
            return None;
        }

        // Advance the seed each batch so different keys are generated each iteration.
        kernel.set_arg(8u32, current_seed).ok()?;
        current_seed = current_seed.wrapping_add(global_size as u64);

        unsafe {
            kernel.enq().ok()?;
        }
        queue.finish().ok()?;
        counter.fetch_add(global_size as u64 * batch_size, Ordering::Relaxed);

        let mut found_host = vec![0i32; 1];
        found_buf.read(&mut found_host).enq().ok()?;

        if found_host[0] != 0 {
            let mut pub_host = vec![0u8; 32];
            let mut priv_host = vec![0u8; 64];
            out_pub.read(&mut pub_host).enq().ok()?;
            out_priv.read(&mut priv_host).enq().ok()?;

            let pubkey_b58 = bs58::encode(&pub_host).into_string();
            let privkey_b58 = bs58::encode(&priv_host).into_string();
            found_flag.store(true, Ordering::Relaxed);
            return Some((pubkey_b58, privkey_b58, priv_host));
        }
    }
}

// ─── SAVE KEYPAIR ────────────────────────────────────────────────────────────

fn save_keypair(path: &str, bytes64: &[u8]) -> std::io::Result<()> {
    let arr: Vec<u8> = bytes64.to_vec();
    let json = serde_json::to_string(&arr).unwrap_or_else(|_| {
        let s: Vec<String> = arr.iter().map(|b| b.to_string()).collect();
        format!("[{}]", s.join(","))
    });
    std::fs::write(path, &json)?;
    println!();
    println!("{}", "✓ Keypair saved.".green().bold());
    println!("  File: {}", path.yellow());
    #[cfg(not(target_os = "windows"))]
    {
        let _ = std::process::Command::new("chmod").args(["600", path]).status();
        println!("  Permissions set to 600 (owner read/write only).");
    }
    println!();
    println!("{}", "⚠  SECURITY WARNING".red().bold());
    println!("  Never share your private key or commit it to version control.");
    println!("  Anyone with access to this file can steal your funds.");
    Ok(())
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

fn format_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else if secs < 86400.0 {
        format!("{:.1}h", secs / 3600.0)
    } else if secs < 86400.0 * 365.0 {
        format!("{:.1}d", secs / 86400.0)
    } else {
        format!("{:.1}y", secs / (86400.0 * 365.25))
    }
}

fn main() {
    let cli = Cli::parse();

    let prefix = cli.prefix.trim().to_string();
    let suffix = cli.suffix.trim().to_string();
    let output = cli.output.clone();
    let case_insensitive = cli.case_insensitive;

    println!();
    println!("{}", "  ◎  Solana Vanity Address Generator".cyan().bold());
    println!("{}", "  ─────────────────────────────────".cyan());
    if !prefix.is_empty() {
        println!("  Prefix : {}", prefix.yellow().bold());
    }
    if !suffix.is_empty() {
        println!("  Suffix : {}", suffix.yellow().bold());
    }
    if prefix.is_empty() && suffix.is_empty() {
        println!("  {}", "No prefix/suffix specified — generating random keypair.".dimmed());
    }
    println!();

    // Detect hardware
    println!("  Detecting hardware...");
    let (logical, _physical, cpu_name) = detect_cpu();
    let threads = cli.threads.unwrap_or(logical);

    #[cfg(feature = "gpu")]
    let gpu_devices = if !cli.cpu_only { detect_gpu() } else { vec![] };
    #[cfg(not(feature = "gpu"))]
    let _gpu_devices: Vec<()> = vec![];

    // Benchmark
    let bench_dur = Duration::from_secs(2);
    println!("  Benchmarking CPU ({} threads, {}s)...", threads, bench_dur.as_secs());
    let cpu_speed = if cli.benchmark || true {
        benchmark_cpu(threads, bench_dur)
    } else {
        0.0
    };

    #[cfg(feature = "gpu")]
    let (selected_gpu, gpu_speed) = if !cli.cpu_only && !gpu_devices.is_empty() {
        let dev = &gpu_devices[0];
        println!("  Benchmarking GPU ({})...", dev.name);
        let spd = benchmark_gpu(dev, bench_dur);
        (Some(dev.name.clone()), spd)
    } else {
        (None, 0.0f64)
    };
    #[cfg(not(feature = "gpu"))]
    let (selected_gpu, gpu_speed): (Option<String>, f64) = (None, 0.0);

    print_hardware_table(&cpu_name, cpu_speed, selected_gpu.as_deref(), gpu_speed);

    // Difficulty estimate
    let pattern_len = prefix.len() + suffix.len();
    if pattern_len > 0 {
        let difficulty = 58f64.powi(pattern_len as i32) / 2.0;
        let effective_speed = if gpu_speed > cpu_speed && !cli.cpu_only {
            gpu_speed
        } else {
            cpu_speed
        };
        let eta_secs = difficulty / effective_speed.max(1.0);
        println!(
            "  Pattern length: {}  |  Difficulty: {:.2e}  |  ETA (avg): {}",
            pattern_len,
            difficulty,
            format_duration(eta_secs).yellow()
        );
        println!();
    }

    if cli.benchmark {
        println!("  Benchmark complete. Exiting.");
        return;
    }

    if prefix.is_empty() && suffix.is_empty() {
        let mut csprng = OsRng;
        let kp = Keypair::generate(&mut csprng);
        let pub_bytes = kp.public.to_bytes();
        let sec_bytes = kp.secret.to_bytes();
        let mut full = Vec::with_capacity(64);
        full.extend_from_slice(&sec_bytes);
        full.extend_from_slice(&pub_bytes);
        let pub_b58 = bs58::encode(&pub_bytes).into_string();
        println!("  Public key: {}", pub_b58.green().bold());
        if let Err(e) = save_keypair(&output, &full) {
            eprintln!("Error saving keypair: {}", e);
        }
        return;
    }

    println!("  {} searching...", "◎".cyan());
    println!();

    let counter = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicBool::new(false));

    // Progress thread
    let counter_prog = counter.clone();
    let found_prog = found.clone();
    let start_prog = Instant::now();
    let _prog_thread = std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            if found_prog.load(Ordering::Relaxed) {
                break;
            }
            let count = counter_prog.load(Ordering::Relaxed);
            let elapsed = start_prog.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed.max(0.001);
            eprint!(
                "\r  Keys tried: {}  |  Rate: {:.0}/s  |  Elapsed: {}    ",
                count,
                rate,
                format_duration(elapsed)
            );
        }
        eprintln!();
    });

    let use_gpu = !cli.cpu_only && gpu_speed > cpu_speed;
    let _ = use_gpu; // suppress warning when gpu feature off

    let mut result: Option<(String, String, Vec<u8>)>;

    #[cfg(feature = "gpu")]
    {
        if use_gpu && selected_gpu.is_some() {
            let dev = &gpu_devices[0];
            result = gpu_worker(dev, &prefix, &suffix, case_insensitive, counter.clone(), found.clone());
            if result.is_none() && !found.load(Ordering::Relaxed) {
                // GPU failed before finding anything — fall back to CPU.
                result = cpu_worker(&prefix, &suffix, case_insensitive, threads, counter.clone(), found.clone());
            }
        } else {
            result = cpu_worker(&prefix, &suffix, case_insensitive, threads, counter.clone(), found.clone());
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        result = cpu_worker(&prefix, &suffix, case_insensitive, threads, counter.clone(), found.clone());
    }

    found.store(true, Ordering::Relaxed);

    if let Some((pubkey, _privkey, bytes64)) = result {
        let elapsed = counter.load(Ordering::Relaxed);
        println!();
        println!("  {} Found!", "◎".green().bold());
        println!("  Public key : {}", pubkey.green().bold());
        println!("  Keys tried : {}", elapsed);
        if let Err(e) = save_keypair(&output, &bytes64) {
            eprintln!("Error saving keypair: {}", e);
            std::process::exit(1);
        }
    } else {
        eprintln!("Search interrupted.");
    }
}
